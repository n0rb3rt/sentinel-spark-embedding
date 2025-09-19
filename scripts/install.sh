#!/bin/bash
set -e

# Parse arguments
ARCH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --arch)
      ARCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--arch cuda|mps|cpu]"
      exit 1
      ;;
  esac
done

# Require architecture for install
if [ -z "$ARCH" ]; then
  echo "Error: --arch is required for installation"
  echo "Usage: $0 --arch cuda|mps|cpu"
  echo "  cuda: For NVIDIA GPU development"
  echo "  mps:  For Apple Silicon GPU development"
  echo "  cpu:  For CPU-only development"
  exit 1
fi

echo "Installing dev dependencies for $ARCH..."

# Bootstrap: install base dependencies first
echo "Bootstrapping base dependencies..."
pip install .

# Cache directory
CACHE_DIR=".cache"
mkdir -p "$CACHE_DIR"/{pip,model/configs}

# Download Clay model checkpoint if not cached
if [ ! -f "$CACHE_DIR/model/clay-v1.5.ckpt" ]; then
  echo "Downloading Clay model checkpoint..."
  curl -L -o "$CACHE_DIR/model/clay-v1.5.ckpt" \
    "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
else
  echo "Using cached Clay model checkpoint"
fi

if [ ! -f "$CACHE_DIR/model/configs/metadata.yaml" ]; then
  echo "Downloading Clay metadata..."
  curl -L -o "$CACHE_DIR/model/configs/metadata.yaml" \
    "https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml"
else
  echo "Using cached Clay metadata"
fi

# Install for development (editable)
if [ "$ARCH" = "cuda" ]; then
  uv pip install --cache-dir "$CACHE_DIR/pip" --extra-index-url https://download.pytorch.org/whl/cu121 -e ".[dev]"
else
  uv pip install --cache-dir "$CACHE_DIR/pip" -e ".[dev]"
fi

# Export Clay model for development architecture if not cached
if [ ! -f "$CACHE_DIR/model/clay-v1.5-encoder-$ARCH.pt2" ]; then
  echo "Exporting Clay model for $ARCH architecture..."
  uv run python -m claymodel.finetune.embedder.factory \
    --img_size 256 \
    --ckpt_path "$CACHE_DIR/model/clay-v1.5.ckpt" \
    --device $ARCH \
    --name "clay-v1.5-encoder-$ARCH.pt2" \
    --ep
  
  # Move compiled model to cache
  mv "checkpoints/compiled/clay-v1.5-encoder-$ARCH.pt2" "$CACHE_DIR/model/"
else
  echo "Using cached Clay model for $ARCH architecture"
fi

echo "Development installation complete!"
echo "Ready for local development and testing."