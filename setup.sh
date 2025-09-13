#!/usr/bin/env bash
#
# Setup script for dinov3-mot
# - Sourcing .env (optional)
# - Finds a compatible Python (3.11/3.12 preferred)
# - Installs basic build deps (Linux/macOS)
# - Creates/activates .venv
# - Installs extras from pyproject.toml:
#     [cpu]    -> torch==2.2.2 (CPU wheels)
#     [t4_gpu] -> torch==2.4.2 (CUDA wheels from ${TORCH_CHANNEL}, default cu124)
# - Optional Hugging Face auth if HUGGINGFACE_HUB_TOKEN is set
# - Optional: Enable **DINOv3** support by upgrading Transformers (stable/edge)
# - Install alias 'venv' -> source .venv/bin/activate
#
set -e

# ---------------- Auto-yes handling ----------------
AUTO_YES=""
DINOV3_MODE=""   # "", "stable", or "edge"
for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES="--yes" ;;
    --dinov3-stable) DINOV3_MODE="stable" ;;
    --dinov3-edge)   DINOV3_MODE="edge" ;;
    cpu|t4_gpu) ;;  # handled later
    *) ;;           # ignore other args here
  esac
done

ask_yes_no() {
  local prompt="$1"
  if [[ -n "$AUTO_YES" ]]; then
    echo "Auto-yes: $prompt -> yes"
    return 0
  fi
  read -p "$prompt " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

# ---------------- Colors & cleanup ----------------
if tput setaf 0 >/dev/null 2>&1; then
  COLOR_GRAY="$(tput setaf 8)"
  COLOR_RESET="$(tput sgr0)"
else
  COLOR_GRAY=$'\033[90m'
  COLOR_RESET=$'\033[0m'
fi

cleanup_render() {
  printf '\r\033[K%s' "${COLOR_RESET}"
  tput cnorm 2>/dev/null || true
}
trap cleanup_render EXIT INT TERM

# ---------------- Error Filtering Function ---------------
handle_error() {
  local description="$1"
  local log_file="$2"

  # Check for specific, known errors and provide clean messages
  if grep -q "Cannot access gated repo" "$log_file"; then
    echo "🔑 Error: Hugging Face authentication failed."
    echo "   This model is restricted and requires a token."
    echo "   To fix, create a .env file with your token and re-run:"
    echo "   echo \"HUGGINGFACE_HUB_TOKEN='hf_...'\" > .env"
  
  # Add more 'elif' checks here for other custom errors in the future
  # elif grep -q "Some other specific error text" "$log_file"; then
  #   echo "🔥 Custom message for another known error."

  else
    # Generic fallback for any unrecognized error
    echo "An unexpected error occurred. Full log below:"
    echo "--- ERROR LOG ---"
    cat "$log_file"
    echo "--- END OF LOG ---"
  fi
}

# ---------------- Spinner run_and_log --------------------

run_and_log() {
  local log_file; log_file=$(mktemp)
  local description="$1"; shift

  # printf "⏳ %s\n" "$description"
  tput civis 2>/dev/null || true

  local prev_render=""
  local cols; cols=$(tput cols 2>/dev/null || echo 120)

  (
    frames=( '⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏' )
    i=0
    while :; do
      local last_line=""
      if [[ -s "$log_file" ]]; then
        last_line=$(tail -n 1 "$log_file" | sed -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g')
      fi

      local prefix="${frames[i]} ${description} : "
      local plain="${prefix}${last_line}"
      if (( ${#plain} > cols )); then
        plain="${plain:0:cols-1}"
      fi

      local head="${plain:0:${#prefix}}"
      local tail="${plain:${#prefix}}"
      local render="${COLOR_RESET}${head}${COLOR_GRAY}${tail}${COLOR_RESET}"

      if [[ "$render" != "$prev_render" ]]; then
        printf '\r\033[K%s' "$render"
        prev_render="$render"
      fi
      i=$(( (i + 1) % ${#frames[@]} ))
      sleep 0.25
    done
  ) &
  local spinner_pid=$!

  if ! "$@" >"$log_file" 2>&1; then
    kill "$spinner_pid" &>/dev/null || true
    wait "$spinner_pid" &>/dev/null || true
    printf '\r\033[K%s' "${COLOR_RESET}"
    printf "❌ %s failed.\n" "$description"

    handle_error "$description" "$log_file"

    rm -f "$log_file"
    exit 1
  fi

  kill "$spinner_pid" &>/dev/null || true
  wait "$spinner_pid" &>/dev/null || true
  printf '\r\033[K%s' "${COLOR_RESET}"
  printf '✅ %s\n' "$description"
  rm -f "$log_file"
}

# ---------------- Step 1: .env ----------------
if [ -f ".env" ]; then
  echo "Sourcing .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
# Bridge legacy token name to HF_TOKEN for Python checks later
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

VENV_DIR=".venv"
PYTHON_BIN=""

# ---------------- Step 2: Python ----------------
echo "Searching for a compatible Python (3.11 or 3.12 preferred)"
if command -v python3.11 &>/dev/null; then
  PYTHON_BIN="python3.11"
elif command -v python3.12 &>/dev/null; then
  PYTHON_BIN="python3.12"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="python3"
  echo "⚠️  Falling back to 'python3' (3.11/3.12 not found)"
else
  echo "❌ No Python interpreter found. Please install Python 3.11 or 3.12."
  exit 1
fi
echo "✅ Using: $($PYTHON_BIN --version)"

# ---------------- Step 3: Build deps ----------------
uname_s="$(uname -s)"
if [[ "$uname_s" == "Linux" ]]; then
  if ! command -v g++ &>/dev/null || ! command -v cmake &>/dev/null; then
    run_and_log "apt-get update" sudo apt-get update
    run_and_log "Install build tools (g++, cmake)" sudo apt-get install -y build-essential g++ cmake
  fi

  if command -v nvidia-smi &>/dev/null && ! command -v nvcc &>/dev/null; then
    echo "NVIDIA GPU detected, but CUDA Toolkit not found."
    if ask_yes_no "Install CUDA Toolkit 12.4 now? [y/N]"; then
    
      run_and_log "Download CUDA keyring" wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
      run_and_log "Install CUDA keyring" sudo dpkg -i cuda-keyring_1.1-1_all.deb

      run_and_log "apt-get update (CUDA)" sudo apt-get update
      rm -f cuda-keyring_1.1-1_all.deb
      run_and_log "Install CUDA toolkit 12-4" sudo apt-get -y install cuda-toolkit-12-4
    fi
  fi

  if [ -d "/usr/local/cuda" ]; then
    export CUDA_PATH="/usr/local/cuda"
    export PATH="${CUDA_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
    if ! grep -q "CUDA_PATH" ~/.profile 2>/dev/null; then
      echo "Adding CUDA to PATH in ~/.profile"
      {
        echo ''
        echo '# CUDA'
        echo "export CUDA_PATH=${CUDA_PATH}"
        echo 'export PATH="${CUDA_PATH}/bin:${PATH}"'
        echo 'export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"'
      } >> ~/.profile
    fi
  fi
elif [[ "$uname_s" == "Darwin" ]]; then
  if ! xcode-select -p &>/dev/null; then
    echo "Xcode Command Line Tools not found. Attempting to install…"
    xcode-select --install || true
  fi
fi

# ---------------- Step 4: venv ----------------
if [ ! -d "$VENV_DIR" ]; then
  run_and_log "Create virtualenv at ${VENV_DIR}" "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Activating virtualenv"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

run_and_log "Upgrade pip" pip install --upgrade pip wheel

# ---------------- Variant selection ----------------
VARIANT=""
for arg in "$@"; do
  case "$arg" in
    cpu|t4_gpu) VARIANT="$arg" ;;
  esac
done

if [[ -z "$VARIANT" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
    echo "Detected GPU: ${gpu_name:-unknown}"
    if [[ -n "$AUTO_YES" ]] || ask_yes_no "Use the GPU install ([t4_gpu])? [y/N]"; then
      VARIANT="t4_gpu"
    else
      VARIANT="cpu"
    fi
  else
    VARIANT="cpu"
  fi
fi

# Enforce runtime-friendly NumPy/SciPy (esp. for torch==2.2.2 CPU)
run_and_log "Ensure NumPy/SciPy ABI compatibility" pip install "numpy<2" "scipy<1.13"

# ---------------- Step 5: install deps (extras) ----------------
TORCH_CHANNEL="${TORCH_CHANNEL:-cu124}"  # for t4_gpu; override if needed (cu121, cu122, cu124)

case "$VARIANT" in
  cpu)
    echo "Installing CPU extras (torch==2.2.2, torchvision==0.17.2)"
    run_and_log "Install base deps [cpu]" pip install -e ".[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
    ;;
  t4_gpu)
    echo "Installing T4 GPU extras (torch==2.4.2, torchvision==0.19.1) from ${TORCH_CHANNEL}"
    run_and_log "Install base deps [t4_gpu]" pip install -e ".[t4_gpu]" --extra-index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
    ;;
  *)
    echo "❌ Unknown variant '$VARIANT'. Use: cpu | t4_gpu"
    exit 1
    ;;
esac

# ---------------- Step 5.5: Enable DINOv3 support (Transformers) ----------------
pick_dinov3_mode() {
  if [[ -n "$DINOV3_MODE" ]]; then
    return
  fi
  if [[ -n "$AUTO_YES" ]]; then
    DINOV3_MODE="stable"   # default when auto-yes
    return
  fi
  echo ""
  if ask_yes_no "Enable DINOv3 support by upgrading Transformers? [y/N]"; then
    echo "Choose mode:"
    echo "  1) stable (PyPI release; recommended)"
    echo "  2) edge   (install from GitHub master)"
    read -p "Enter 1 or 2 [1]: " choice
    case "$choice" in
      2) DINOV3_MODE="edge" ;;
      *) DINOV3_MODE="stable" ;;
    esac
  else
    DINOV3_MODE=""
  fi
}

install_dinov3_support() {
  case "$DINOV3_MODE" in
    stable)
      run_and_log "Install Transformers (stable) for DINOv3" \
        pip install -U "transformers>=4.45" "huggingface_hub>=0.25"
      ;;
    edge)
      run_and_log "Install Transformers (edge) for DINOv3" \
        pip install -U "git+https://github.com/huggingface/transformers.git" "huggingface_hub>=0.25"
      ;;
    *)
      echo "Skipping Transformers upgrade (DINOv3 may remain unsupported)."
      return 0
      ;;
  esac

  # Verify DINOv3 is recognized (lightweight check; no large weights download)
  run_and_log "Verify DINOv3 support in Transformers" \
    python - <<'PY'
import os, sys
from transformers import AutoConfig, AutoModel
repo = "facebook/dinov3-vitb16-pretrain-lvd1689m"
tok  = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
kw   = {}
try:
    AutoConfig.from_pretrained(repo, token=tok)
    kw["token"] = tok
except TypeError:
    # transformers < 4.44 used use_auth_token
    if tok: kw["use_auth_token"] = tok

cfg = AutoConfig.from_pretrained(repo, **kw)
print("model_type:", cfg.model_type)
# This will fail if the library doesn't map dinov3_vit -> a modeling class
try:
    _ = AutoModel.from_config(cfg)
except Exception as e:
    print("dinov3_supported: False")
    raise
else:
    print("dinov3_supported: True")
PY
}

pick_dinov3_mode
install_dinov3_support

# ---------------- Step 6: Hugging Face auth (optional) ----------------
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  run_and_log "Hugging Face auth" huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN" --add-to-git-credential
fi

# ---------------- Step 7: Aliases ----------------
alias venv='source .venv/bin/activate'

if [ -f ~/.bashrc ]; then
    echo "alias venv='source .venv/bin/activate'" >> ~/.bashrc
    echo "✅ Alias 'venv' added to your Bash shell."
fi

if [ -f ~/.zshrc ]; then
    echo "alias venv='source .venv/bin/activate'" >> ~/.zshrc
    echo "✅ Alias 'venv' added to your Zsh shell."
fi

echo ""
echo "✅ Setup complete (variant: ${VARIANT}${DINOV3_MODE:+, dinov3=${DINOV3_MODE}})."
echo "Activate:   venv or source ${VENV_DIR}/bin/activate"
echo "Run demo:   python cli.py --source data/input.mp4 --output outputs/tracked.mp4"
