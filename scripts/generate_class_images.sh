#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

CLASS="dog"
PROMPT="A $CLASS"
GUIDANCE_SCALE=7.5
NUM_IMAGES=200
MODEL_NAME="runwayml/stable-diffusion-v1-5"


python src/generate_images.py \
--pretrained_model $MODEL_NAME \
--guidance_scale $GUIDANCE_SCALE \
--num_images $NUM_IMAGES \
--prompt "$PROMPT" \
--infer_steps 20 \
--save_path "data/generated_$CLASS" \
--gpu 0