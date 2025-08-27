# two_stage_pipeline.py
import os
import torch
from torchvision import transforms
from PIL import Image
from helper import load_pipeline
from controlnet_aux import CannyDetector
from diffusers.utils import load_image, make_image_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":

    # load pipeline with subject LoRAs
    pipeline = load_pipeline(
        pipeline_type='txt2img',
        subject_ckpt_dir="./dreambooth_lora_ckpt",
        lora_unet_dir="./dreambooth_lora_ckpt/unet_lora_epoch1",
        lora_text_dir="./dreambooth_lora_ckpt/text_lora_epoch1",
        device = device
    )

    # --- Stage 1: txt->img (compose subject in scene)

    prompt = "A photo of <vobj> in a natural setting, waterfall in the background"
    stage_a = pipeline(prompt = prompt, 
                #    image=..., 
                num_inference_steps = 30, 
                strength = 0.5).images[0]
    stage_a.save("./stageA_comp.png")

    # --- Stage 2 : img->img (add style to subject-image)
    pipeline = load_pipeline(
        pipeline_type='img2img',
        device=device
    )

    canny = CannyDetector()
    canny_img = canny(stage_a, detect_resolution=768, image_resolution=768)

    ip_adap_img = load_image("style_refs/vangogh/The Church at Auvers.jpg")

    images = pipeline(prompt = "A photo", 
                negative_prompt = "low quality",
                height = 768, 
                width = 768,
                ip_adapter_image = ip_adap_img,
                image = canny_img,
                guidance_scale = 6,
                controlnet_conditioning_scale = 0.7,
                num_inference_steps = 20,
                num_images_per_prompt = 3).images

    images = [(stage_a.resize((768, 768)))] + images

    out = make_image_grid(images, rows = 1, cols = 4)

    out.save("./stageB_final.png")
    print("Saved final image")
