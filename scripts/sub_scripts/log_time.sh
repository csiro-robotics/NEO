#!/bin/bash

# Function to measure command execution time and log to a specified JSON file
log_time() {
    local log_file="$1" # First argument is the log file path
    local label="$2"    # Second argument is the command label
    shift 2             # Remove the first two arguments from the positional parameters

    # Ensure the directory exists
    local dir=$(dirname "$log_file")
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"   # Create the directory if it doesn't exist
    fi

    # If the log file doesn't exist, create it as an empty JSON object
    if [[ ! -f "$log_file" ]]; then
        echo "{}" > "$log_file"
    fi

    local start_time=$(date +%s.%N) # Start time in seconds.nanoseconds
    "$@"                           # Execute the command
    local end_time=$(date +%s.%N)   # End time in seconds.nanoseconds
    local elapsed=$(echo "$end_time - $start_time" | bc) # Calculate elapsed time

      # Read, update, and write back to the JSON file
    jq --arg label2 "$label" --arg elapsed2 "$elapsed" \
        '. + {($label2): ($elapsed2 | tonumber)}' "$log_file" > "$log_file.tmp"

    # Check if jq succeeded, then move the temporary file and delete it
    if [[ $? -eq 0 ]]; then
        mv "$log_file.tmp" "$log_file" && rm -f "$log_file.tmp" # Move and delete temp file
    else
        echo "Error updating the log file. Temporary file not deleted."
    fi
}