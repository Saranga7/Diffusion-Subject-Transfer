#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

LORA_MODEL="dreambooth_lora_ckpt_dog"
GUIDANCE_SCALE=5
PROMPT="A photo of <vobj> dog in the style of Van Gogh"

python src/infer_subject.py \
--pretrained_model_name "runwayml/stable-diffusion-v1-5" \
--lora_path "$LORA_MODEL/" \
--guidance_scale $GUIDANCE_SCALE \
--prompt "$PROMPT" \
--infer_steps 30 \
--save_path "ouput_$GUIDANCE_SCALE" \
--gpu 0