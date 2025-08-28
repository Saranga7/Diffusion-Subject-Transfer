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
   parser.add_argument("--pretrained_model_name", type = str, default = "runwayml/stable-diffusion-v1-5")
   parser.add_argument('--lora_path', type = str, required = True)
   parser.add_argument('--prompt', type = str, required = True)
   parser.add_argument('--guidance_scale', type = float, default = 7.5)
   parser.add_argument('--infer_steps', type = int, default = 30)
   parser.add_argument('--save_path', type = str, default='out')
   parser.add_argument('--gpu', type = int, default = 1)
   # parser.add_argument('--seed', type = int, default = 999)
   
   # parser.add_argument('--seed', type = int, default = 7)
   args = vars(parser.parse_args())

   device = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"

   pipeline = StableDiffusionPipeline.from_pretrained(args['pretrained_model_name'], safety_checker = None)
   pipeline.to(device)

   # replace tokenizer with fine-tuned tokenizer
   pipeline.tokenizer = CLIPTokenizer.from_pretrained(args['lora_path'])
   print(len(pipeline.tokenizer))
   # replace text encoder with base model then load text LoRA adapter
   text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

   # resizing encoder to account for the new subject token
   text_encoder.resize_token_embeddings(len(pipeline.tokenizer))

   unet_lora_path = os.path.join(args['lora_path'], "unet_lora_epoch7")
   text_lora_path = os.path.join(args['lora_path'], "text_lora_epoch7")

   pipeline.text_encoder = PeftModel.from_pretrained(text_encoder, text_lora_path)
   pipeline.unet = PeftModel.from_pretrained(pipeline.unet, unet_lora_path)

   # gen = torch.Generator(device=device)
   # gen.manual_seed(args['seed'])

   # prompt = "A photo of <vobj> in space"
   style_images = pipeline(prompt = args['prompt'], 
                  num_inference_steps = args['infer_steps'],
                  guidance_scale = args['guidance_scale'],
                  num_images_per_prompt = 3,
                  # generator = gen
                  ).images

   out = make_image_grid(style_images, rows = 1, cols = 3)

   out.save(f"{args['save_path']}.png")


if __name__ == "__main__":
   main()

