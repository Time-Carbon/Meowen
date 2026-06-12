#!/usr/bin/bash

rank=(32 64 96 128 160)

for r in ${rank[@]};do
    python main.py \
    --lora_rank $r \
    $@ &> "train_rank$r.log"
done