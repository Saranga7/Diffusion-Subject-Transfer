#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$pwd"

LORA_MODEL="dreambooth_lora_ckpt"
STRENGTH=0.5
PROMPT="A photo of <vobj> in front of the Eiffel Tower"

python src/infer_subject.py \
--lora_path "$LORA_MODEL/" \
--strength $STRENGTH \
--prompt "$PROMPT" \
--infer_steps 30 \
--save_path "output" \