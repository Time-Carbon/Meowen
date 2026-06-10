#!/usr/bin/bash

rank=(8 16 32 64 128 256)

for r in ${rank[@]};do
    python main.py \
    --lora_rank $r \
    $@ &> "train_rank$r.log"
done