#!/bin/bash

# -------------------------------------------------------
# Script: transfer.sh
# Description:
#   Copies the latest realsense_recording_* folder to an
#   external HDD under: /media/robotics/One Touch/SPARC-Data/April_Pilot
#   with folder naming: MO_serial_DDMM
# -------------------------------------------------------

set -e

SRC_BASE=~/realsense_recording_*
DEST_BASE="/media/robotics/One Touch/SPARC-Data/April_Pilot"

# Detect latest recording folder
latest_folder=$(ls -td $SRC_BASE 2>/dev/null | head -1)

if [ ! -d "$latest_folder" ]; then
  echo "[ERROR] No realsense_recording_* folder found!"
  exit 1
fi

echo "[INFO] Latest recording folder detected: $latest_folder"

# Get today's date
DDMM=$(date +%d%m)
MO=$(date +%b | cut -c1-2 | tr '[:lower:]' '[:upper:]') # e.g., AP

# Ask for serial number (default to 01 if confirmed)
read -p "[?] Use serial number '01' for this folder name? (y/N): " use_default
if [[ "$use_default" =~ ^[Yy]$ ]]; then
  serial="01"
else
  read -p "[?] Enter 2-digit serial number (e.g., 02): " serial
fi

folder_name="${MO}_${serial}_${DDMM}"
destination="$DEST_BASE/$folder_name"
mkdir -p "$destination"

echo "[INFO] Destination: $destination"
echo "[INFO] 🔄 Copying files using cp -av..."

# Track failures
declare -a failed_files=()

# Generate list of files to copy
files_to_copy=$(find "$latest_folder" -type f)

for src_file in $files_to_copy; do
  relative_path="${src_file#$latest_folder/}"
  dest_file="$destination/$(basename "$latest_folder")/$relative_path"

  mkdir -p "$(dirname "$dest_file")"

  for attempt in {1..3}; do
    cp -v "$src_file" "$dest_file"
    sync

    src_hash=$(sha256sum "$src_file" | awk '{print $1}')
    dest_hash=$(sha256sum "$dest_file" | awk '{print $1}')

    if [[ "$src_hash" == "$dest_hash" ]]; then
      echo "[✅] Copied: $relative_path"
      break
    else
      echo "[WARN] ❌ Checksum mismatch for $relative_path (attempt $attempt)"
      sleep 2
    fi

    if [[ $attempt -eq 3 ]]; then
      echo "[ERROR] ❌ Failed to copy $relative_path after 3 attempts."
      failed_files+=("$relative_path")
    fi
  done
done

echo "[✅] All matching files copied and verified."

# 🔍 Summary of failures
if [ ${#failed_files[@]} -ne 0 ]; then
  echo ""
  echo "[❌ SUMMARY] The following files failed to copy after retries:"
  for f in "${failed_files[@]}"; do
    echo "  - $f"
  done
  echo ""
else
  echo "[✅] All files verified successfully!"
fi

echo "[🏁 DONE] Transfer to $destination complete."
