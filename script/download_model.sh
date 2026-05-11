#!/bin/bash

modelscope download --model Qwen/Qwen3.5-4B-Base --local_dir ./model
modelscope download --token $TOKEN --dataset ABABA123/porn_text_raw --local_dir ./dataset