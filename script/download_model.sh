#!/bin/bash

modelscope download --model ABABA123/Qwen3.5-0.8B-Base-bnb-8bit --local_dir ./model
modelscope download --token $TOKEN --dataset ABABA123/porn_text_raw --local_dir ./dataset