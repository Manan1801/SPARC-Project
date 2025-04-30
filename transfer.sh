#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# Script: transfer.sh
# Description:
#   Handles SPARC-Project data transfer:
#   Mode 0 - Archive latest recording folder, transfer, verify, extract, deep verify
#   Mode 1 - Copy each file individually with SHA-256 verify
# -------------------------------------------------------

# -------------------- Logging helpers --------------------
log_info()  { echo -e "🟢 [INFO]  $*"; }
log_warn()  { echo -e "🟡 [WARN]  $*"; }
log_error() { echo -e "🔴 [ERROR] $*"; }

# -------------------- Timer helpers --------------------
timer_display() {
    local s=$SECONDS
    printf "⏱ Elapsed %02d:%02d:%02d" $((s/3600)) $((s%3600/60)) $((s%60))
}

start_timer() {
    SECONDS=0
    ( while true; do
        printf "\r"; timer_display
        sleep 1
    done ) &
    TIMER_PID=$!
}

stop_timer() {
    if [[ -n "${TIMER_PID-}" ]]; then
        kill "$TIMER_PID" &>/dev/null || true
        unset TIMER_PID
        echo
    fi
}

# -------------------- Prompt Mode --------------------
echo "Select transfer type:"
echo "  0) Frames (archive → verify → extract → verify)"
echo "  1) Bag files (copy each file individually)"
while true; do
    read -rp "Enter 0 or 1: " MODE
    [[ $MODE =~ ^[01]$ ]] && break
    echo "Please enter 0 or 1."
done

# -------------------- Detect Latest Recording Folder --------------------
SRC_BASE=~/realsense_recording_*
latest_folder=$(ls -td $SRC_BASE 2>/dev/null | head -1)
if [[ ! -d $latest_folder ]]; then
    log_error "No realsense_recording_* folder found!"
    exit 1
fi
log_info "Latest folder: $latest_folder"

# -------------------- Setup Destination --------------------
DEST_BASE="/media/robotics/One Touch/SPARC-Data/April_Pilot"
DDMM=$(date +%d%m)
MO=$(date +%b | cut -c1-2 | tr '[:lower:]' '[:upper:]')  # e.g., AP
read -rp "[?] Use serial '01'? (y/N): " yn
if [[ $yn =~ ^[Yy]$ ]]; then
    serial="01"
else
    read -rp "[?] Enter 2-digit serial (e.g., 02): " serial
fi
folder="${MO}_${serial}_${DDMM}"
destination="$DEST_BASE/$folder"
mkdir -p "$destination"
log_info "Destination: $destination"

# -------------------- Function: Transfer Single File with Verify --------------------
failures=()
copy_with_verify() {
    local src="$1" dst="$2"
    mkdir -p "$(dirname "$dst")"
    for attempt in {1..3}; do
        log_info "🚚 Copying $(basename "$src") (Attempt $attempt/3)"
        start_timer
        rsync -a --info=progress2 "$src" "$dst" 2>&1 | while read -r line; do
            if [[ $line =~ ([0-9]+)% ]]; then
                printf "\r"; timer_display
                printf " | 📊 %s%%" "${BASH_REMATCH[1]}"
            fi
        done
        stop_timer

        local src_sum dst_sum
        src_sum=$(sha256sum "$src" | awk '{print $1}')
        dst_sum=$(sha256sum "$dst" | awk '{print $1}')
        if [[ "$src_sum" == "$dst_sum" ]]; then
            log_info "✅ Verified $(basename "$src")"
            return 0
        else
            log_warn "⚠️ Checksum mismatch for $(basename "$src")"
            sleep 2
        fi
    done
    log_error "❌ Failed $(basename "$src") after 3 attempts"
    failures+=("$src")
    return 1
}

# -------------------- Mode 0: Archive Transfer --------------------
if [[ $MODE == "0" ]]; then
    archive_name="/tmp/$(basename "$latest_folder").tar"
    log_info "📦 Creating archive $archive_name"
    tar -cf "$archive_name" -C "$(dirname "$latest_folder")" "$(basename "$latest_folder")"

    src_sum=$(sha256sum "$archive_name" | awk '{print $1}')
    log_info "🔍 SHA256 of local archive: $src_sum"

    # Copy archive to destination
    dst_archive="$destination/$(basename "$archive_name")"
    copy_with_verify "$archive_name" "$dst_archive"

    # Verify destination archive
    dst_sum=$(sha256sum "$dst_archive" | awk '{print $1}')
    if [[ "$src_sum" != "$dst_sum" ]]; then
        log_error "❌ Archive verification failed even after retries!"
        exit 1
    fi

    # Extract archive
    log_info "📂 Extracting archive in destination..."
    start_timer
    tar -xf "$dst_archive" -C "$destination"
    stop_timer
    log_info "✅ Extraction completed."

    # Deep verification of extracted files
    log_info "🔍 Verifying extracted files..."
    for file in $(find "$latest_folder" -type f); do
        relative_path="${file#$latest_folder/}"
        src_file="$file"
        dst_file="$destination/$(basename "$latest_folder")/$relative_path"
        if [[ ! -f "$dst_file" ]]; then
            log_error "❌ Missing file after extraction: $relative_path"
            failures+=("$relative_path")
            continue
        fi
        src_hash=$(sha256sum "$src_file" | awk '{print $1}')
        dst_hash=$(sha256sum "$dst_file" | awk '{print $1}')
        if [[ "$src_hash" != "$dst_hash" ]]; then
            log_error "❌ Checksum mismatch after extraction: $relative_path"
            failures+=("$relative_path")
        fi
    done

    # Delete tar files
    log_info "🧹 Deleting archives..."
    rm -f "$archive_name" "$dst_archive"

# -------------------- Mode 1: File-by-File Transfer --------------------
else
    log_info "📂 Starting file-by-file copy..."
    find "$latest_folder" -type f | while read -r src_file; do
        relative_path="${src_file#$latest_folder/}"
        dst_file="$destination/$(basename "$latest_folder")/$relative_path"
        copy_with_verify "$src_file" "$dst_file"
    done
fi

# -------------------- Summary --------------------
if (( ${#failures[@]} )); then
    log_error "❌ Summary of failed transfers:"
    for f in "${failures[@]}"; do
        echo "   - $f"
    done
else
    log_info "🎉 All transfers and verifications completed successfully!"
fi

log_info "🏁 DONE!"
