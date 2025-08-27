# helpers.py (put somewhere you can import)
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPTokenizer, CLIPTextModel
from peft import PeftModel

def load_pipeline(pipeline_type, 
                    subject_ckpt_dir = None, 
                    lora_unet_dir = None, 
                    lora_text_dir = None, 
                    device='cuda',
                    ):
    # load base pipeline (txt2img / image2image)
    if pipeline_type == 'txt2img':
        assert subject_ckpt_dir is not None, "subject_ckpt_dir must be provided for txt2img"
        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    else:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", 
            torch_dtype=torch.float16,
            varient="fp16").to(device)
    

        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "Yntec/AbsoluteReality",
            #"runwayml/stable-diffusion-v1-5",
            controlnet=controlnet, 
            torch_dtype=torch.float16).to(device)

        pipeline.load_ip_adapter("h94/IP-Adapter", 
                     subfolder = "models", 
                     weight_name = "ip-adapter_sd15.bin")
        pipeline.set_ip_adapter_scale(0.3)

        # pipeline.enable_model_cpu_offload()

        return pipeline

    # replace tokenizer with trained tokenizer (contains the new token)
    pipeline.tokenizer = CLIPTokenizer.from_pretrained(subject_ckpt_dir)

    # load base text encoder and then wrap it with text LoRA if present
    pipeline.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    if lora_text_dir:
        pipeline.text_encoder = PeftModel.from_pretrained(pipeline.text_encoder, lora_text_dir).to(device)

    # wrap unet with subject LoRA if present
    if lora_unet_dir:
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, lora_unet_dir).to(device)

    return pipeline
