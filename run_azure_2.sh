#!/bin/bash

# Define the list of tasks
tasks=("psychology" "robotics")

# Define the model and cache folder
model="azure_openai"
model_cache_folder="/gpfs/radev/scratch/cohan/jw3278/azure_openai"

# Loop through each task and run the command
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python run.py --task "$task" --model "$model" --model_cache_folder "$model_cache_folder"
    echo "Finished task: $task"
    echo "--------------------------------------------"
done
