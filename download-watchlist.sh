#!/bin/bash

# URL of the file to download
FILE_URL="http://zenscans.com/watchlist/watchlist.zip"

# Folder to save the downloads
DOWNLOAD_DIR="/Users/steven/projects/ZenBot/ZenBot.Screener.Web/Client/wwwroot/watchlist"

# Create the directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Loop forever
while true; do
    # Generate a timestamped filename
    TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
    OUTPUT_FILE="$DOWNLOAD_DIR/watchlist.zip"
    ARCHIVE_FILE="$DOWNLOAD_DIR/watchlist-$TIMESTAMP.zip"

    # Download the file
    curl -L "$FILE_URL" -o "$OUTPUT_FILE"
    # Save a timestamped copy
    cp "$OUTPUT_FILE" "$ARCHIVE_FILE"

    # Keep only the 10 most recent archived files
    # List archived files by modification time (newest first) and remove old ones
    ls -t "$DOWNLOAD_DIR"/watchlist-*.zip 2>/dev/null | tail -n +11 | xargs -r rm -f

    # Wait 60 seconds
    sleep 60
done
