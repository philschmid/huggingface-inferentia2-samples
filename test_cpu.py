from transformers import AutoModelForCausalLM, AutoTokenizer
# Load and save the CPU model
model_cpu = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6b', low_cpu_mem_usage=True, revision='sharded')
# Get a tokenizer and exaple input
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
text = "Hello, I'm a language model,"
encoded_input = tokenizer(text, return_tensors='pt', padding=True)

# Run inference using temperature
sample_output = model_cpu.generate(
    input_ids=encoded_input.input_ids,
    attention_mask=encoded_input.attention_mask,
    do_sample=True,
    max_length=256,
    temperature=0.7,
)
print([tokenizer.decode(tok) for tok in sample_output])