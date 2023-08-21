from optimum.neuron.version import __version__

print(f"Running on {__version__}")
from optimum.neuron import NeuronStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
input_shapes = {"batch_size": 1, "sequence_length": 64, "height": 512, "width": 512}

stable_diffusion = NeuronStableDiffusionPipeline.from_pretrained(
    model_id,
    export=True,
    **input_shapes,
    # device_ids=[0, 1],
)

# Save locally or upload to the HuggingFace Hub
save_directory = "sd_neuron/"
stable_diffusion.save_pretrained(save_directory)

# COMMENT IN TO UPLOAD TO HUB
# stable_diffusion.push_to_hub(
#      save_directory, repository_id="my-neuron-repo", use_auth_token=True
# )
