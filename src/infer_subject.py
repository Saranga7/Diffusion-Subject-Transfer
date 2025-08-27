from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid
from transformers import CLIPTokenizer, CLIPTextModel
from peft import PeftModel
import torch
import os
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--lora_path', type = str, required = True)
   parser.add_argument('--prompt', type = str, required = True)
   parser.add_argument('--strength', type = float, default = 0.5)
   parser.add_argument('--infer_steps', type = int, default = 30)
   parser.add_argument('--save_path', type = str, default='out')
   # parser.add_argument('--seed', type = int, default = 7)
   args = vars(parser.parse_args())
   pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
   pipeline.to(device)

   # replace tokenizer with fine-tuned tokenizer
   pipeline.tokenizer = CLIPTokenizer.from_pretrained(args['lora_path'])
   print(len(pipeline.tokenizer))
   # replace text encoder with base model then load text LoRA adapter
   text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

   # resizing encoder to account for the new subject token
   text_encoder.resize_token_embeddings(len(pipeline.tokenizer))

   # unet_lora_path = "./dreambooth_lora_ckpt/unet_lora_epoch1"
   # text_lora_path = "./dreambooth_lora_ckpt/text_lora_epoch1"

   unet_lora_path = os.path.join(args['lora_path'], "unet_lora_epoch1")
   text_lora_path = os.path.join(args['lora_path'], "text_lora_epoch1")

   pipeline.text_encoder = PeftModel.from_pretrained(text_encoder, text_lora_path)
   pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_lora_path)



   # prompt = "A photo of <vobj> in space"
   style_images = pipeline(prompt = args['prompt'], 
                  num_inference_steps = args['infer_steps'],
                  strength = args['strength'],
                  num_images_per_prompt = 3,
                  ).images

   out = make_image_grid(style_images, rows = 1, cols = 3)

   out.save(f"{args['save_path']}.png")


if __name__ == "__main__":
   main()

