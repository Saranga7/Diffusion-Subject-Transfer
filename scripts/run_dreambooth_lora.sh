#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"
IDENTIFIER="<vobj>"
CLASS="dog"

start_time=$(date +%s)

python src/train_dreambooth_lora.py \
  --pretrained_model "runwayml/stable-diffusion-v1-5" \
  --instance_data_dir "data/Dali_images" \
  --class_data_dir "data/generated_dog" \
  --instance_token $IDENTIFIER \
  --class_word $CLASS \
  --class_prompt "a $CLASS" \
  --instance_prompt "a $IDENTIFIER $CLASS" \
  --output_dir ./dreambooth_lora_ckpt_dog \
  --num_epochs 30 \
  --batch_size 2 \
  --lr 1e-4 \
  --class_preservation_weight 1.0 \
  --gpu 0


end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Elapsed time: $((elapsed/60)) minutes and $((elapsed%60)) seconds"