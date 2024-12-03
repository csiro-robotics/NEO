#!/bin/bash

# Base directory where you want to search for timestamped folders (can be modified)
base_dir="outputs"

# Loop through all directories under base_dir
find "$base_dir" -type d | while read -r dir; do
  # Skip if it's the base directory itself (don't want to delete anything in the root)
  if [[ "$dir" == "$base_dir" ]]; then
    continue
  fi

  # Find all directories inside this directory
  dirs=$(find "$dir" -type d)

  # If no directories are found, skip this directory
  if [[ -z "$dirs" ]]; then
    continue
  fi

  # Now, search for timestamped directories using the correct regex with -regextype posix-egrep
  timestamp_dirs=$(echo "$dirs" | find "$dir" -type d -regextype posix-egrep -regex '.*[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}')

  # If no timestamp directories are found, skip
  if [[ -z "$timestamp_dirs" ]]; then
    continue
  fi

  # Sort directories based on timestamp in folder names (YYYY-MM-DD_HHMMSS)
  newest_dir=$(echo "$timestamp_dirs" | sort -t'_' -k1,1 -k2,2 -r | head -n 1)

  # Remove all directories except the newest one in this specific folder
  for dir in $timestamp_dirs; do
    if [[ "$dir" != "$newest_dir" ]]; then
      # Print the removed directory
      echo "Removed: $dir"
      rm -rf "$dir"
    fi
  done

  # Print the saved (newest) directory
  echo "Saved: $newest_dir"
done

echo "Cleanup process completed."
