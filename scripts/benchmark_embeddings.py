# primary Script used for inf2 Benchmarking

import time
import torch
import neuronperf
import neuronperf.torch
import argparse
import torch_neuronx
from transformers import (
    AutoModel, AutoTokenizer  # Any other model class respective to the model we want to infer on
)

# pip install neuronperf --extra-index-url=https://pip.repos.neuron.amazonaws.com
# 

class MeanPoolEncoder(torch.nn.Module):
    def __init__(self, model_id, **kwargs):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, **kwargs)

    def forward(self, input_ids, attention_mask,token_type_ids=None) -> torch.Tensor:
        # forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # mean pooling
        token_embeddings = outputs[0] #First element of model_output contains all token embeddings
        sentence_embeddings = self.mean_pooling(token_embeddings, attention_mask)
        return sentence_embeddings

    def mean_pooling(self,token_embeddings, attention_mask, token_type_ids=None):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def create_dummy_sample(batch_size, sequence_length, tokenizer):
    dummy = "This is a sample"
    dummy = [dummy] * batch_size
    tokenized_sample = tokenizer(dummy, padding="max_length", max_length=sequence_length, return_tensors="pt")
    return tuple(tokenized_sample.values())


def benchmark(model_name, batch_size, sequence_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MeanPoolEncoder(model_name, torchscript=True)
    model.eval()

    example = create_dummy_sample(batch_size, sequence_length, tokenizer)

    res = model(*example)
    assert res.shape == (batch_size, 768), "Unexpected output shape"

    start_time = time.time()
    traced = torch_neuronx.trace(model, example,compiler_args=["--auto-cast", "all", "--auto-cast-type", "bf16"])
    print(f"Tracing took {time.time() - start_time} seconds")
    filename = f'model_bs{batch_size}_seq{sequence_length}.pt'
    torch.jit.save(traced, filename)
    reports = neuronperf.torch.benchmark(filename, [example], [batch_size], cost_per_hour=0.76)
    # View and save results
    print("======== {} ========".format(filename))
    neuronperf.print_reports(reports)
    neuronperf.write_json(reports,filename=f"results_bs{batch_size}_seq{sequence_length}.json")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='BAAI/bge-base-en-v1.5')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sequence_length', type=int, default=128)
    
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    # benchmark(model_name, batch_size, sequence_length)
    # Below are a few examples -
    args = parse_args()
    print(args)
    
    benchmark(**vars(args))
    # benchmark('bert-base-uncased', 4, 128)
    # benchmark('gpt2', 16, 256)