# generate_class_images.py
import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm


def generate(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model,
        safety_checker=None,
    )
    pipeline.to(device)
    # pipeline.unet.to(device)
    # pipeline.text_encoder.to(device)

    out_dir = Path(args.save_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    prompt = args.prompt
    for i in tqdm(range(args.num_images)):
        out = pipeline(
            prompt,
            num_inference_steps=args.infer_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
        )
        img = out.images[0]
        img.save(out_dir / f"class_{i:04d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type = str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type = str, default="A dog", help="class prompt to generate images for")
    parser.add_argument("--num_images", type = int, default = 200)
    parser.add_argument("--save_path", type = str, default="./generated_class_images")
    parser.add_argument("--infer_steps", type = int, default = 20)
    parser.add_argument("--guidance_scale", type=float, default = 7.5)
    parser.add_argument("--height", type = int, default = 512)
    parser.add_argument("--width", type = int, default = 512)
    parser.add_argument("--gpu", type = int, default = 1)
    args = parser.parse_args()
    generate(args)

    print(f"Generated {args.num_images} images in {args.save_path}")
