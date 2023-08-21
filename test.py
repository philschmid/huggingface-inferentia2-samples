from optimum.neuron import NeuronStableDiffusionPipeline

model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
# model_id = "optimum/tiny-stable-diffusion-neuronx"
input_shapes = {"batch_size": 1, "sequence_length": 18, "height": 512, "width": 512}

stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(
    model_id, export=True, device_ids=[0, 1], **input_shapes
)
# stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(model_id, device_ids=[0, 1])

prompt = "sailing ship in storm by Leonardo da Vinci"
image = stable_diffusion(prompt).images[0]