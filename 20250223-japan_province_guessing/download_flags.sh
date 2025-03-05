#!/usr/bin/env bash

# set -eo pipefail

# https://commons.wikimedia.org/wiki/Flags_of_prefectures_of_Japan

# Create directory to store flags
mkdir -p flag_svgs

echo "Start"

# Read the JSON file and extract prefecture names, then download flags
jq -r '.[].name_en' prefecture_infos.json | while read -r prefecture; do
    # Construct the Wikimedia URL
    url="https://commons.wikimedia.org/wiki/File:Flag_of_${prefecture// /_}_Prefecture.svg"
    
    # Download the Wikimedia page
    wget -qO page.html "$url"
    
    # Extract the actual file download URL
    svg_url=$(grep fullMedia page.html | grep -oE 'https://upload.wikimedia.org[^" ]+\.svg' | head -n 1)
    
    if [[ -n "$svg_url" ]]; then
        # Download the SVG file
        wget -qO "flag_svgs/${prefecture}.svg" "$svg_url"
        echo "Downloaded: ${prefecture}.svg"
    else
        echo "Failed parsing flag url for: $prefecture"
    fi

done

# Clean up
rm -f page.html
