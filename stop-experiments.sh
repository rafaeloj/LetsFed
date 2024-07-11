#!/bin/bash

# Define the file with PIDs
pid_file="/tmp/docker_compose_pids.txt"

# Check if the file exists and is not empty
if [[ -s "$pid_file" ]]; then
    # Loop through each PID in the file and kill the process
    while IFS= read -r pid; do
        echo "Killing process with PID: $pid"
        kill "$pid"
    done < "$pid_file"

    # Optionally, remove the pid_file after killing the processes
    rm "$pid_file"
else
    echo "No PIDs found to kill."
fi

echo "All specified processes have been killed."
