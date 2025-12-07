#!/bin/bash

# Batch video dubbing script with night-time scheduling
# Usage: ./batch_dubbing.sh <input_dir> <output_dir>

set -e  # Exit on any error

# Function to display usage
usage() {
    echo "Usage: $0 <input_dir> <output_dir>"
    echo "  input_dir  - Directory containing MP4 videos to process"
    echo "  output_dir - Directory where dubbed videos will be saved"
    exit 1
}

# Function to check if it's night time (22:00-03:00)
is_night_time() {
    local current_hour=$(date +%H)
    # Convert to integer to avoid octal interpretation
    current_hour=$((10#$current_hour))
    
    # Night time is from 22:00 to 03:00
    if [[ $current_hour -ge 22 || $current_hour -lt 3 ]]; then
        return 0  # It's night time
    else
        return 1  # It's day time
    fi
}

# Function to wait for night time
wait_for_night() {
    while ! is_night_time; do
        local current_time=$(date +%H:%M)
        echo "Current time: $current_time - Waiting for night time (22:00-03:00)..."
        echo "Next check in 30 minutes..."
        sleep 300  # Wait 5 minutes before checking again
    done
    echo "Night time detected! Starting processing..."
}

# Check if correct number of arguments provided
if [[ $# -ne 2 ]]; then
    echo "Error: Invalid number of arguments"
    usage
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Validate input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Convert to absolute paths
INPUT_DIR=$(realpath "$INPUT_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Find all MP4 files in input directory and store in array
declare -a mp4_files
while IFS= read -r -d $'\0' file; do
    mp4_files+=("$file")
done < <(find "$INPUT_DIR" -name "*.mp4" -type f -print0)

if [[ ${#mp4_files[@]} -eq 0 ]]; then
    echo "No MP4 files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#mp4_files[@]} MP4 files to process"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Process each MP4 file
for video_file in "${mp4_files[@]}"; do
    # Extract filename without path and extension, handle special characters
    filename=$(basename "$video_file")
    filename="${filename%.mp4}"
    output_file="$OUTPUT_DIR/${filename}_ru.mp4"
    
    # Skip if output file already exists
    if [[ -f "$output_file" ]]; then
        echo "Skipping $filename - output file already exists: $output_file"
        continue
    fi
    
    echo "=================================================="
    echo "Processing: $filename"
    echo "Input: $video_file"
    echo "Output: $output_file"
    echo "=================================================="
    
    # Wait for night time if not already night
    if ! is_night_time; then
        echo "Not night time - waiting for 22:00-03:00 window..."
        wait_for_night
    fi
    
    # Run the dubbing command
    echo "Starting dubbing process at $(date)..."
    
    if python dubblm_cli.py \
        --input "$video_file" \
        --output "$output_file" \
        --source_language en \
        --target_language ru; then
        
        echo "✅ Successfully processed: $filename"
        echo "Output saved to: $output_file"
    else
        echo "❌ Failed to process: $filename"
        echo "Continuing with next file..."
    fi
    
    echo "Completed processing $filename at $(date)"
    echo ""
done

echo "=================================================="
echo "Batch processing completed!"
echo "Processed files are in: $OUTPUT_DIR"
echo "==================================================" 