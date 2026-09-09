set -e

echo "RunPod GPU Setup for Ollama Pipeline"

echo ""
echo "[1/7] Verifying GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU drivers not installed."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "✓ GPUs detected"

echo ""
echo "[2/7] Stopping existing Ollama instances..."
pkill -9 ollama || true
sleep 2
echo "✓ Cleared existing Ollama processes"

echo ""
echo "[3/7] Configuring Ollama GPU settings..."

export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_LAYERS=35
export OLLAMA_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

cat << 'EOF' >> ~/.bashrc
export OLLAMA_NUM_GPU=1
export OLLAMA_GPU_LAYERS=35
export OLLAMA_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1
EOF

echo "✓ GPU configuration set"
echo "  - OLLAMA_NUM_GPU: 1"
echo "  - OLLAMA_GPU_LAYERS: 35"
echo "  - OLLAMA_NUM_THREADS: 4"
echo "  - GPU Assignment: GPU 1 (Ollama), GPU 0 (YOLO)"

echo ""
echo "[4/7] Starting Ollama server on GPU 1..."
CUDA_VISIBLE_DEVICES=1 nohup ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
sleep 5

if ps -p $OLLAMA_PID > /dev/null; then
    echo "✓ Ollama server started (PID: $OLLAMA_PID)"
else
    echo "ERROR: Failed to start Ollama server"
    cat /tmp/ollama.log
    exit 1
fi

echo ""
echo "[5/7] Pulling quantized LLaVA model (llava:7b)..."
ollama pull llava:7b

if ollama list | grep -q "llava:7b"; then
    echo "✓ Quantized LLaVA model ready"
else
    echo "ERROR: Failed to pull quantized model"
    exit 1
fi

echo ""
echo "[6/7] Verifying GPU allocation..."
sleep 3

if nvidia-smi | grep -q "ollama"; then
    echo "✓ Ollama is using GPU"
    nvidia-smi | grep ollama
else
    echo "WARNING: Ollama not detected in nvidia-smi"
    echo "Check /tmp/ollama.log for details"
fi

echo ""
echo "[7/7] Updating config.json..."

cp config.json config.json.backup

python3 << 'PYEOF'
import json

with open('config.json', 'r') as f:
    config = json.load(f)

config['ollama_config']['ollama_model'] = 'llava:7b'

if 'prompt_files' not in config['ollama_config']:
    config['ollama_config']['prompt_files'] = {}

config['ollama_config']['prompt_files']['extended_visibility_all'] = 'extended_visibility_all.txt'

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ config.json updated")
PYEOF

echo ""
echo "*** GPU Setup Complete! ***"
echo ""
echo "Configuration Summary:"
echo "  - Ollama server: Running on GPU 1 (PID: $OLLAMA_PID)"
echo "  - Model: llava:7b (quantized)"
echo "  - GPU Layers: 35"
echo "  - YOLO will use: GPU 0"
echo "  - Processing: Sequential (no parallel threads)"
echo ""
echo "Next Steps:"
echo "  1. Copy extended_visibility_all.txt to data/prompts/"
echo "  2. Run your pipeline with: CUDA_VISIBLE_DEVICES=0 python main.py"
echo "  3. Monitor GPU usage with: watch -n 1 nvidia-smi"
echo ""
echo "Logs:"
echo "  - Ollama: /tmp/ollama.log"
echo "  - Pipeline: outputs/pipeline.log"
echo ""
echo "To stop Ollama: pkill ollama"
