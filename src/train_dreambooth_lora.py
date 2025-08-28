# train_dreambooth_lora.py
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

from dataset import DreamBoothDataset



def train(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name,
        safety_checker = None,   
    )
    pipeline.to(device)
    pipeline.enable_attention_slicing()  # reduce memory

    # Add new token
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    new_token = args.instance_token  # e.g. "<my_subj>"
    class_word = args.class_word     # e.g. "person"

    if new_token in tokenizer.get_vocab():
        print(f"Token {new_token} already exists in tokenizer.")
    else:
        tokenizer.add_tokens(new_token)
        print(f"Added token {new_token} to tokenizer.")
        text_encoder.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer.convert_tokens_to_ids(new_token)
    print("new token id:", token_id)


    # Initialize new token embedding with embedding of class_word tokens averaged
    init_ids = tokenizer(class_word, add_special_tokens=False).input_ids
    with torch.no_grad():
        emb = text_encoder.get_input_embeddings().weight.data
        if len(init_ids) > 0:
            emb[token_id] = emb[init_ids].mean(dim=0).clone()
            print(f"Initialized {new_token} from `{class_word}` tokens: {init_ids}")

    # Wrap UNet and text_encoder with LoRA
    lora_cfg_unet = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = args.unet_target_modules.split(','),
        lora_dropout = args.lora_dropout,
        bias = 'none',
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_cfg_unet)
    print("Wrapped UNet with LoRA")

    lora_cfg_text = LoraConfig(
        r = args.lora_r_text,
        lora_alpha = args.lora_alpha_text,
        target_modules = args.text_target_modules.split(','),
        lora_dropout = args.lora_dropout_text,
        bias = 'none',
    )
    pipeline.text_encoder = get_peft_model(pipeline.text_encoder, lora_cfg_text)
    print("Wrapped text_encoder with LoRA")

    
    pipeline.to(device)
    pipeline.unet.to(device)
    pipeline.text_encoder.to(device)
    pipeline.vae.to(device)




    DB_dataset = DreamBoothDataset(
        instance_folder = args.instance_data_dir,
        class_folder = args.class_data_dir,
        image_size = args.image_size
    )
    dl = DataLoader(DB_dataset, 
                    batch_size = args.batch_size, 
                    shuffle = True, 
                    num_workers = args.num_workers, 
                    drop_last = True)

    # Only train LoRA parameters
    trainable_params = [p for p in list(pipeline.unet.parameters()) + list(pipeline.text_encoder.parameters()) if p.requires_grad]
    print("Trainable params count:", sum([p.numel() for p in trainable_params]))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # scheduler
    global_step = 0
    for epoch in range(args.num_epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for inst_imgs, class_imgs in pbar:
            batch_size = inst_imgs.shape[0]
            inst_imgs = inst_imgs.to(device)
            class_imgs = class_imgs.to(device)

            # Build prompts for the batch
            inst_prompts = [args.instance_prompt for _ in range(batch_size)]
            class_prompts = [args.class_prompt for _ in range(batch_size)]

            # Tokenize prompts
            inst_tokens = tokenizer(inst_prompts, padding = True, return_tensors="pt").to(device)
            class_tokens = tokenizer(class_prompts, padding = True, return_tensors="pt").to(device)

            # Encode text -> hidden states
            inst_hidden = pipeline.text_encoder(**inst_tokens).last_hidden_state
            class_hidden = pipeline.text_encoder(**class_tokens).last_hidden_state

            # VAE encode images to latents
            with torch.no_grad():
                inst_latents = pipeline.vae.encode(inst_imgs).latent_dist.sample() * 0.18215 # Scales the latent vector by 0.18215 so it’s in the correct range for the diffusion model.
                class_latents = pipeline.vae.encode(class_imgs).latent_dist.sample() * 0.18215 

            # Sample noise & timesteps
            noise_inst = torch.randn_like(inst_latents)
            noise_class = torch.randn_like(class_latents)
            timesteps = torch.randint(0, int(1000 * args.train_strength), (batch_size,), device=device, dtype=torch.long)

            noisy_inst = pipeline.scheduler.add_noise(inst_latents, noise_inst, timesteps)
            noisy_class = pipeline.scheduler.add_noise(class_latents, noise_class, timesteps)

            # UNet predictions
            noise_pred_inst = pipeline.unet(noisy_inst, timesteps, encoder_hidden_states = inst_hidden).sample
            noise_pred_class = pipeline.unet(noisy_class, timesteps, encoder_hidden_states = class_hidden).sample

            # compute MSE
            loss_inst = F.mse_loss(noise_pred_inst, noise_inst)
            loss_class = F.mse_loss(noise_pred_class, noise_class)
            loss = loss_inst + args.class_preservation_weight * loss_class

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": float(loss.detach().cpu())})

            # Save intermediate checkpoints
            if global_step % args.save_interval == 0:
                save_checkpoints(pipeline, tokenizer, args.output_dir, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

        # epoch end save
        save_checkpoints(pipeline, tokenizer, args.output_dir, step=f"epoch{epoch+1}")

    # Final save
    save_checkpoints(pipeline, tokenizer, args.output_dir, step="final")
    print("Training finished. Saved to:", args.output_dir)


def save_checkpoints(pipeline, tokenizer, out_dir, step = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    # Save tokenizer
    tokenizer.save_pretrained(out_dir)
    # Save LoRA adapters for unet and text encoder
    unet_out = out_dir / f"unet_lora_{step}"
    text_out = out_dir / f"text_lora_{step}"
    pipeline.unet.save_pretrained(unet_out)
    pipeline.text_encoder.save_pretrained(text_out)
    print(f"Saved adapters: {unet_out}, {text_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name", type = str, default = "runwayml/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type = str, required = True, help = "Folder with subject images")
    parser.add_argument("--class_data_dir", type = str, required = True, help = "Folder with class images (for prior preservation)")
    parser.add_argument("--instance_token", type = str, default = "<my_subject>", help = "New token to learn")
    parser.add_argument("--class_word", type = str, default = "person", help = "Class word (e.g., 'person', 'dog')")
    parser.add_argument("--instance_prompt", type = str, required = True)
    parser.add_argument("--class_prompt", type = str, required = True)

    parser.add_argument("--image_size", type = int, default = 512)
    parser.add_argument("--batch_size", type = int, default = 2)
    parser.add_argument("--num_epochs", type = int, default = 50)

    parser.add_argument("--max_train_steps", type = int, default = 2000)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--train_strength", type = float, default = 0.3)
    parser.add_argument("--class_preservation_weight", type = float, default = 1.0)
    parser.add_argument("--lora_r", type = int, default = 16)
    parser.add_argument("--lora_alpha", type = int, default = 32)
    parser.add_argument("--lora_dropout", type = float, default = 0.05)
    parser.add_argument("--unet_target_modules", type = str, default = "to_k,to_q,to_v,to_out.0,proj_out", help = "comma-separated unet target module names")
    parser.add_argument("--lora_r_text", type = int, default = 8)
    parser.add_argument("--lora_alpha_text", type = int, default = 16)
    parser.add_argument("--lora_dropout_text", type = float, default = 0.05)
    parser.add_argument("--text_target_modules", type = str, default = "q_proj,k_proj,v_proj,out_proj", help = "comma-separated text encoder target modules")
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--output_dir", type = str, required = True)
    parser.add_argument("--save_interval", type = int, default = 500)
    parser.add_argument("--gpu", type = int, default = 1)
    args = parser.parse_args()
    train(args)
