#!/bin/bash
# Download 4K HDRI files from Poly Haven
# Run from the project root directory

set -e

HDRI_DIR="static/hdri"
mkdir -p "$HDRI_DIR"

# HDRI names (matching the files referenced in script.js)
HDRIS=(
  "studio_small_03"
  "venice_sunset"
  "industrial_workshop_foundry"
  "brown_photostudio_02"
  "modern_buildings"
  "kloppenheim_02"
  "autumn_forest"
  "blue_photo_studio"
  "courtyard"
  "kiara_dawn"
  "shanghai_bund",
  "spiaggia_di_mondello"
  "symmetrical_garden_02"
)

echo "Downloading 4K HDRI files from Poly Haven..."
echo ""

for name in "${HDRIS[@]}"; do
  output="$HDRI_DIR/${name}.hdr"
  url="https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/${name}_4k.hdr"

  if [ -f "$output" ]; then
    echo "[$name] File exists, downloading new version..."
  fi

  echo "[$name] Downloading 4K version..."
  if curl -L --fail -o "$output" "$url" 2>/dev/null; then
    size=$(ls -lh "$output" | awk '{print $5}')
    echo "[$name] Downloaded successfully ($size)"
  else
    echo "[$name] Failed to download 4K, trying 2K..."
    url_2k="https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/${name}_2k.hdr"
    if curl -L --fail -o "$output" "$url_2k" 2>/dev/null; then
      size=$(ls -lh "$output" | awk '{print $5}')
      echo "[$name] Downloaded 2K version ($size)"
    else
      echo "[$name] FAILED - Could not download"
    fi
  fi
  echo ""
done

echo "Done! HDRI files are in $HDRI_DIR/"
echo ""
echo "File sizes:"
ls -lh "$HDRI_DIR"/*.hdr 2>/dev/null || echo "No HDR files found"
