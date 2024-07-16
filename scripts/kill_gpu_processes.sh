#!/bin/bash

bold=$(tput bold)
normal=$(tput sgr0)

# Run nvidia-smi to get the list of processes using the GPU
echo -e "Running \`nvidia-smi\` and killing all GPU processes..."
output=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)

# Check if the output is not empty
if [[ -z "$output" ]]; then
    echo "...${bold}No GPU processes found.${normal}"
    exit 0
fi

# Loop through each PID and kill the process
for pid in $output; do
    echo "...Killing PID: $pid"
    kill -9 $pid
done

echo "${bold}All GPU processes have been killed.${normal}"
