#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Run name (customize as needed)
run="bs32epoch100"

# Python script
python_script="run.py"

# Create logs directory if it doesn't exist
mkdir -p logs

# Dataset names and corresponding config files
datasets=("PSM" "MSL" "SMAP" "SMD" "SWAT")
configs=("psm.yaml" "msl.yaml" "smap.yaml" "smd.yaml" "swat.yaml")

# Run for each dataset
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    config=${configs[$i]}
    log_file="logs/${dataset}_${run}.log"

    echo "=== Running dataset: $dataset with config: $config ==="
    echo "Logging to: $log_file"

    # Run the command and save both stdout and stderr to log file
    python "$python_script" --dataset "$dataset" --config "$config" --run "$run" 2>&1 | tee "$log_file"

    echo "=== Finished dataset: $dataset ==="
    echo
done

echo "All datasets have finished running. Logs are saved in the logs/ directory."
