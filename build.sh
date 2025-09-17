#!/bin/bash
set -e

# Parse arguments
ARCH=""
DEV_MODE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dev)
      DEV_MODE=true
      shift
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dev] [--arch cuda|mps|cpu]"
      exit 1
      ;;
  esac
done

# Cache directory
CACHE_DIR=".cache"
mkdir -p "$CACHE_DIR"/{pip,venvs,model/configs}

# Calculate hash of pyproject.toml for venv caching
DEPS_HASH=$(sha256sum pyproject.toml | cut -d' ' -f1 | head -c12)
VENV_CACHE="$CACHE_DIR/venvs/base-venv-${DEPS_HASH}.tar.gz"

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

# Auto-detect architecture if not provided
if [ -z "$ARCH" ]; then
  ARCH=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
fi

echo "Using architecture: $ARCH"

if [ "$DEV_MODE" = true ]; then
  # Dev mode: just install locally, skip dist creation
  echo "Installing dev dependencies locally for $ARCH..."
  if [ "$ARCH" = "cuda" ]; then
    uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e ".[dev]"
  else
    uv pip install -e ".[dev]"
  fi
else
  # Production mode: create distribution assets
  rm -rf dist
  mkdir -p dist/{venv,lib,model,src}
  
  # Build wheel using uv (handles build dependencies automatically)
  echo "Building wheel..."
  uv build --wheel --out-dir dist/lib
  
  # Install dependencies locally (non-editable for production)
  echo "Installing build dependencies locally for $ARCH..."
  if [ "$ARCH" = "cuda" ]; then
    uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 ".[build]"
  else
    uv pip install ".[build]"
  fi
  
  # Build base venv for distribution if not cached
  if [ ! -f "$VENV_CACHE" ]; then
    echo "Building distributable venv: $DEPS_HASH"
    
    python3.12 -m venv /tmp/pack-env
    source /tmp/pack-env/bin/activate
    
    # Use uv consistently for all installations
    if [ "$ARCH" = "cuda" ]; then
      /tmp/pack-env/bin/python -m pip install uv
      /tmp/pack-env/bin/uv pip install --cache-dir "$CACHE_DIR/pip" \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        .[build]
    else
      /tmp/pack-env/bin/python -m pip install uv
      /tmp/pack-env/bin/uv pip install --cache-dir "$CACHE_DIR/pip" .[build]
    fi
    
    # Remove our project package, keep only dependencies
    /tmp/pack-env/bin/uv pip uninstall sentinel-processing
    
    # Pack the base venv
    venv-pack -o "$VENV_CACHE"
  else
    echo "Using cached distributable venv: $DEPS_HASH"
  fi
fi

echo "Building Clay model for architecture: $ARCH"

# Export Clay model for target architecture if not cached
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

if [ "$DEV_MODE" = true ]; then
  echo "Dev build complete - ready for local development!"
else
  # Copy model files to dist
  cp "$CACHE_DIR/model/clay-v1.5-encoder-$ARCH.pt2" "dist/model/"
  cp -r "$CACHE_DIR/model/configs" "dist/model/"
  
  # Link cached venv to dist
  ln "$VENV_CACHE" "dist/venv/venv.tar.gz"
  
  # Copy source code
  cp -r src/* dist/src/
  
  # Clean up old venv caches
  find "$CACHE_DIR/venvs" -name "base-venv-*.tar.gz" ! -name "base-venv-${DEPS_HASH}.tar.gz" -delete 2>/dev/null || true
  
  echo "Build complete:"
  echo "  Environment: dist/venv/venv.tar.gz (hash: $DEPS_HASH)"
  echo "  Libraries: dist/lib/*.whl"
  echo "  Source: dist/src/"
  echo "  Model: dist/model/clay-v1.5-encoder-$ARCH.pt2"
fi

# Clean up build artifacts
find . \( -name "__pycache__" -o -name "*.egg-info" -o -name "build" \) -exec rm -rf {} + 2>/dev/null || true