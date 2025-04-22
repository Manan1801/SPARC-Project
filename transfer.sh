#!/bin/bash

# -----------------------------------------------------------
# Transfer latest RealSense recording to external HDD with
# integrity checks and backup folder naming
#
# Destination: /media/robotics/One\ Touch/SPARC-Data/April_Pilot/
# Folder format: <MO_serial_DDMM> e.g., AP_01_1904
# -----------------------------------------------------------

# ----- Config -----
DEST_ROOT="/media/robotics/One Touch/SPARC-Data/April_Pilot"
MONTH_CODE="AP"  # Change this if needed for other months

# ----- Step 1: Find latest recording folder -----
latest_folder=$(ls -td ~/realsense_recording_* 2>/dev/null | head -n 1)

if [ -z "$latest_folder" ]; then
  echo "[❌ ERROR] No recording folder found under ~/"
  exit 1
fi

echo "[INFO] Latest recording folder detected: $latest_folder"

# ----- Step 2: Extract DDMM from timestamp -----
basename=$(basename "$latest_folder")  # realsense_recording_YYYYMMDD_HHMMSS
datetime=${basename#"realsense_recording_"}
date_part=${datetime%%_*}  # YYYYMMDD
ddmm="${date_part:6:2}${date_part:4:2}"  # DDMM

# ----- Step 3: Ask user if serial number should auto-increment -----
read -p "[?] Use serial number '01' for this folder name? (y/N): " yn

if [[ "$yn" =~ ^[Yy]$ ]]; then
  serial="01"
else
  read -p "[?] Enter custom serial number (e.g., 02, 03...): " serial
fi

# ----- Step 4: Format destination folder name as MO_serial_DDMM -----
dest_folder_name="${MONTH_CODE}_${serial}_${ddmm}"
dest_path="$DEST_ROOT/$dest_folder_name"

echo "[INFO] Destination: $dest_path"

# ----- Step 5: Create destination directory -----
mkdir -p "$dest_path" || { echo "[❌ ERROR] Failed to create destination."; exit 1; }

# ----- Step 6: Start copying files -----
echo "[INFO] 🔄 Copying files using cp -av..."
cp -av "$latest_folder"/ "$dest_path"/

echo "[✅] Copy complete."

# ----- Step 7: Generate SHA256 checksums -----
echo "[INFO] 🔍 Verifying checksums..."
src_hash="$latest_folder/.sha256sum.txt"
dst_hash="$dest_path/.sha256sum.txt"

# Generate source checksums
find "$latest_folder" -type f -exec sha256sum {} + | sort > "$src_hash"
# Generate destination checksums
find "$dest_path" -type f -exec sha256sum {} + | sort > "$dst_hash"

# Compare
if diff "$src_hash" "$dst_hash" > /dev/null; then
  echo "[✅] SHA256 Verification passed! Files are identical."
  rm "$src_hash" "$dst_hash"
else
  echo "[❌ WARNING] SHA256 MISMATCH! Some files may be corrupted."
  echo "Check the diff between:"
  echo "  $src_hash"
  echo "  $dst_hash"
fi

echo "[🏁 DONE] Transfer and verification process finished."
