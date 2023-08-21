import os
# To use two neuron core per worker
os.environ["NEURON_RT_NUM_CORES"] = "2"
import torch
import torch_neuronx
import base64
from io import BytesIO
from optimum.neuron import NeuronStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler


def model_fn(model_dir):
    # load local converted model into pipeline
    input_shapes = {"batch_size": 1,"sequence_length":64, "height": 512, "width": 512}

    pipeline = NeuronStableDiffusionPipeline.from_pretrained(model_dir, device_ids=[0, 1],**input_shapes)
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline


def predict_fn(data, pipeline):
    prompt = data.pop("inputs", data)
    # set valid HP for stable diffusion
    num_inference_steps = data.pop("num_inference_steps", 25)
    guidance_scale = data.pop("guidance_scale", 7.5)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)

    # pass inputs with all kwargs in data
    generated_images = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        # can be extended
    )["images"]
    
    # postprocess convert image into base64 string
    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # always return the first 
    return {"generated_images": encoded_images}

pipeline = model_fn("sd_neuron")

images = predict_fn({"inputs":" A cat"}, pipeline)
print(images)