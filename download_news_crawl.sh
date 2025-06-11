#!/bin/bash

# Download script for news-crawl documents from data.statmt.org
# Usage: ./download_news_crawl.sh [start_year] [end_year] [num_parallel]

BASE_URL="https://data.statmt.org/news-crawl/doc/en/"
DOWNLOAD_DIR="./data/news-crawl-data"

# Default values
START_YEAR=${1:-2007}
END_YEAR=${2:-2024}
NUM_PARALLEL=${3:-3}  # Number of parallel downloads

# Create download directory
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "=========================================="
echo "News-Crawl Data Download Script"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo "Download directory: $DOWNLOAD_DIR"
echo "Year range: $START_YEAR - $END_YEAR"
echo "Parallel downloads: $NUM_PARALLEL"
echo "=========================================="

# Function to download a single file
download_file() {
    local year=$1
    local filename="news-docs.${year}.en.filtered.gz"
    local url="${BASE_URL}${filename}"
    
    echo "Starting download: $filename"
    
    # Use wget with resume capability, timeout, and retry options
    wget \
        --continue \
        --timeout=30 \
        --tries=3 \
        --retry-connrefused \
        --progress=bar:force \
        --show-progress \
        "$url"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded: $filename"
        # Verify file integrity by checking if it's a valid gzip file
        if gzip -t "$filename" 2>/dev/null; then
            echo "✓ File integrity verified: $filename"
        else
            echo "✗ File integrity check failed: $filename"
        fi
    else
        echo "✗ Failed to download: $filename"
    fi
}

# Export function for parallel execution
export -f download_file
export BASE_URL

# Generate list of years to download
years=()
for year in $(seq $START_YEAR $END_YEAR); do
    years+=($year)
done

echo "Files to download: ${#years[@]}"
echo "Years: ${years[*]}"
echo ""

# Download files in parallel using xargs
printf '%s\n' "${years[@]}" | xargs -n 1 -P "$NUM_PARALLEL" -I {} bash -c 'download_file "$@"' _ {}

echo ""
echo "=========================================="
echo "Download Summary"
echo "=========================================="

# Check downloaded files
total_size=0
successful_downloads=0
failed_downloads=0

for year in "${years[@]}"; do
    filename="news-docs.${year}.en.filtered.gz"
    if [ -f "$filename" ]; then
        size=$(du -h "$filename" | cut -f1)
        echo "✓ $filename - $size"
        # Add to total size (convert to bytes for accurate calculation)
        size_bytes=$(stat -c%s "$filename" 2>/dev/null || stat -f%z "$filename" 2>/dev/null)
        total_size=$((total_size + size_bytes))
        ((successful_downloads++))
    else
        echo "✗ $filename - NOT FOUND"
        ((failed_downloads++))
    fi
done

# Convert total size to human readable
total_size_human=$(numfmt --to=iec-i --suffix=B $total_size 2>/dev/null || echo "$total_size bytes")

echo ""
echo "Total files downloaded: $successful_downloads"
echo "Failed downloads: $failed_downloads"
echo "Total size: $total_size_human"
echo "Download directory: $(pwd)"

if [ $failed_downloads -eq 0 ]; then
    echo "✓ All downloads completed successfully!"
else
    echo "⚠ Some downloads failed. You can re-run the script to retry failed downloads."
fi 