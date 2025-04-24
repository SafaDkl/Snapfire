import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import warnings
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_default_dtype(torch.float32)

def generate_edge_map(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.Canny(blurred, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

def apply_scanner_darkly_ai(input_path, output_path, prompt, negative_prompt, steps, seed):
    
    # Load ControlNet model for edge guidance
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

    # Load base Stable Diffusion pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    safety_checker=None
)

    # Optimize performance
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cpu")

    # Prepare input image and edge map
    init_image = load_image(input_path).resize((512, 512))
    control_image = generate_edge_map(input_path).resize((512, 512))
    
    # Generate image
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
        image=control_image,
        height=512,
        width=512,
    ).images[0]

    # Save outputf
    result.save(output_path)
    
# Example usage
apply_scanner_darkly_ai('input.jpg', 'output.jpg', "apply scanner darkly theme on the input image", "nature, scenery", 30, 0)
