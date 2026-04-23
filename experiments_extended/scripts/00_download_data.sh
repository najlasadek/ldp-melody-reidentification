#!/usr/bin/env bash
# =============================================================================
# 00_download_data.sh
# Download MTC-FS-INST-2.0 (Meertens Tune Collections) and SymbTr datasets.
# Run from the experiments_extended/ directory:
#   bash scripts/00_download_data.sh
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$BASE_DIR/data"

echo "=== Data download script ==="
echo "Data directory: $DATA_DIR"

# =============================================================================
# Dataset 1: Meertens Tune Collections (MTC-FS-INST-2.0)
# Source: https://zenodo.org/records/3551003
# Format: ZIP containing JSON Lines file(s) with melodic feature sequences
# Pitch sequences already extracted as MIDI integers — no music parser needed.
# =============================================================================

MEERTENS_DIR="$DATA_DIR/meertens_raw"
MEERTENS_ZIP="$MEERTENS_DIR/MTC-FS-INST-2.0.zip"

echo ""
echo "--- Meertens Tune Collections (MTC-FS-INST-2.0) ---"
echo "Source: https://zenodo.org/records/3551003"

if [ -d "$MEERTENS_DIR" ] && [ "$(ls -A "$MEERTENS_DIR" 2>/dev/null | grep -v '^$')" ]; then
    echo "Already exists: $MEERTENS_DIR (skipping download)"
else
    mkdir -p "$MEERTENS_DIR"
    echo "Downloading MTC-FS-INST-2.0..."

    # Try wget first, fall back to curl
    if command -v wget &>/dev/null; then
        wget -O "$MEERTENS_ZIP" \
            "https://zenodo.org/records/3551003/files/MTC-FS-INST-2.0.zip?download=1" \
            --progress=bar:force 2>&1
    elif command -v curl &>/dev/null; then
        curl -L -o "$MEERTENS_ZIP" \
            "https://zenodo.org/records/3551003/files/MTC-FS-INST-2.0.zip?download=1" \
            --progress-bar
    else
        echo "ERROR: Neither wget nor curl found. Please install one and retry."
        exit 1
    fi

    echo "Extracting..."
    cd "$MEERTENS_DIR"
    unzip -q "$MEERTENS_ZIP"
    echo "Meertens download complete."
fi

echo "Meertens directory contents:"
ls -la "$MEERTENS_DIR/"

# =============================================================================
# Dataset 2: SymbTr — Turkish Makam Music
# Source: https://github.com/MTG/SymbTr
# We download ONLY the MIDI files (midi/ subfolder).
# Pitch-bend is NEVER used. Only note_on pitch integers (0-127) are extracted.
# =============================================================================

SYMBTR_DIR="$DATA_DIR/symbtr_raw"

echo ""
echo "--- SymbTr Turkish Makam Music (MIDI files only) ---"
echo "Source: https://github.com/MTG/SymbTr"

if [ -d "$SYMBTR_DIR" ] && [ "$(find "$SYMBTR_DIR" -name '*.mid' 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "Already exists with MIDI files: $SYMBTR_DIR (skipping download)"
else
    mkdir -p "$SYMBTR_DIR"

    # Option A: git clone (preferred — gets just what we need)
    if command -v git &>/dev/null; then
        echo "Cloning SymbTr repository (MIDI files only via sparse checkout)..."
        cd "$SYMBTR_DIR"

        git init
        git remote add origin https://github.com/MTG/SymbTr.git

        # Sparse checkout to get only MIDI files (saves bandwidth)
        git config core.sparseCheckout true
        echo "midi/" >> .git/info/sparse-checkout

        echo "Fetching (this may take a few minutes for ~2200 MIDI files)..."
        git pull origin master --depth=1

        echo "SymbTr clone complete."

    elif command -v wget &>/dev/null; then
        # Option B: Download zip from GitHub
        echo "Downloading SymbTr ZIP from GitHub..."
        SYMBTR_ZIP="$DATA_DIR/symbtr_raw.zip"
        wget -O "$SYMBTR_ZIP" \
            "https://github.com/MTG/SymbTr/archive/refs/heads/master.zip" \
            --progress=bar:force 2>&1
        echo "Extracting..."
        unzip -q "$SYMBTR_ZIP" -d "$DATA_DIR/"
        mv "$DATA_DIR/SymbTr-master" "$SYMBTR_DIR" 2>/dev/null || true
        rm -f "$SYMBTR_ZIP"
        echo "SymbTr download complete."

    elif command -v curl &>/dev/null; then
        echo "Downloading SymbTr ZIP from GitHub..."
        SYMBTR_ZIP="$DATA_DIR/symbtr_raw.zip"
        curl -L -o "$SYMBTR_ZIP" \
            "https://github.com/MTG/SymbTr/archive/refs/heads/master.zip" \
            --progress-bar
        echo "Extracting..."
        unzip -q "$SYMBTR_ZIP" -d "$DATA_DIR/"
        mv "$DATA_DIR/SymbTr-master" "$SYMBTR_DIR" 2>/dev/null || true
        rm -f "$SYMBTR_ZIP"
        echo "SymbTr download complete."

    else
        echo "ERROR: Neither git, wget, nor curl found."
        echo "Please manually download SymbTr MIDI files from:"
        echo "  https://github.com/MTG/SymbTr"
        echo "and place them in: $SYMBTR_DIR"
        exit 1
    fi
fi

echo ""
MIDI_COUNT=$(find "$SYMBTR_DIR" -name "*.mid" 2>/dev/null | wc -l)
echo "SymbTr MIDI files found: $MIDI_COUNT"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=== Download complete ==="
echo "Next step: run  python scripts/01_prepare_datasets.py"
echo ""
echo "Directory summary:"
echo "  $MEERTENS_DIR"
ls -la "$MEERTENS_DIR/" 2>/dev/null | head -10
echo "  $SYMBTR_DIR"
ls -la "$SYMBTR_DIR/" 2>/dev/null | head -5
