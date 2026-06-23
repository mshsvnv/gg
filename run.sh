#!/bin/bash

set -e

# python -m pip install -r req.txt

# echo "Generate flow..."
# python generate_flow.py

#MODELS=("IFRNet" "IFRNet_RGB2" "IFRNet_YUV420")
MODELS=("IFRNet_RGB2P")

for MODEL in "${MODELS[@]}"; do

    #echo "Starting experiment for: $MODEL"
    #python train.py --model_name "$MODEL" --device  cuda:0

    echo "  Benchmarking $MODEL..."
    python benchmarks/speed_parameters.py --model_name "$MODEL"> "./checkpoint/${MODEL}/speed_parameters.txt"

    echo "  Evaluating $MODEL..."
    python benchmarks/eval_vimeo90k.py --model_name "$MODEL" > "./checkpoint/${MODEL}/eval.txt"

    echo "Finished experiment for: $MODEL"
done

echo "All experiments completed successfully!"
