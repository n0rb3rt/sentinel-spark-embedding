#!/bin/bash
set -e

# Auto-detect architecture from existing environment
ARCH=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
echo "Detected architecture: $ARCH"

# Verify install.sh was run
if [ ! -f ".cache/model/clay-v1.5-encoder-$ARCH.pt2" ]; then
  echo "Error: Run './scripts/install.sh --arch $ARCH' first to set up development environment"
  exit 1
fi

# Cache directory
CACHE_DIR=".cache"
mkdir -p "$CACHE_DIR/venvs"

# Calculate hash of pyproject.toml for venv caching
DEPS_HASH=$(sha256sum pyproject.toml | cut -d' ' -f1 | head -c12)
VENV_CACHE="$CACHE_DIR/venvs/base-venv-${DEPS_HASH}.tar.gz"

# Production build: create distribution assets
rm -rf dist
mkdir -p dist/{venv,lib,model,src}

# Build wheel using uv (handles build dependencies automatically)
echo "Building wheel..."
uv build --wheel --out-dir dist/lib

# Build base venv for distribution if not cached
if [ ! -f "$VENV_CACHE" ]; then
  echo "Building distributable venv: $DEPS_HASH"
  
  # Create clean copy of dev environment
  python3.12 -m venv /tmp/pack-env
  cp -r .venv/lib/python3.12/site-packages/* /tmp/pack-env/lib/python3.12/site-packages/
  source /tmp/pack-env/bin/activate
  
  # Remove PySpark and our project package, keep other dependencies
  /tmp/pack-env/bin/uv pip uninstall pyspark sentinel-processing
  
  # Pack the base venv
  venv-pack -o "$VENV_CACHE"
else
  echo "Using cached distributable venv: $DEPS_HASH"
fi

# Copy cached model files to dist
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

# Clean up build artifacts
find . \( -name "__pycache__" -o -name "*.egg-info" -o -name "build" \) -exec rm -rf {} + 2>/dev/null || true