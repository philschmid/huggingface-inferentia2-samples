import os
from transformers_neuronx.gptj.model import GPTJForSampling
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'
# Load and save the CPU model
split_dir='gptj-split'
model_id='EleutherAI/gpt-j-6b'
revision='sharded'

####### LOAD AND COMPILE THE MODEL #######
model_cpu = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, revision=revision)
save_pretrained_split(model_cpu, split_dir)

# Create and compile the Neuron model
model = GPTJForSampling.from_pretrained(split_dir, batch_size=1, tp_degree=2, n_positions=512, amp='f32', unroll=None)
model.to_neuron()
model = HuggingFaceGenerationModelAdapter(model_cpu.config, model)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'



def model_fn(model_dir):
    return model, tokenizer

def predict_fn(data, model_tokenizer):
    model, tokenizer = model_tokenizer
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", None)

    # preprocess
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids

    # pass inputs with all kwargs in data
    model.reset_generation()
    if parameters is not None:
        outputs = model.generate(input_ids, **parameters)
    else:
        outputs = model.generate(input_ids, do_sample=True, temperature=0.7)

    # postprocess the prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return [{"generated_text": prediction}]

model_tokenizer = model_fn(None)

print(predict_fn({"inputs": "Hello, I'm a language model,"}, model_tokenizer))
