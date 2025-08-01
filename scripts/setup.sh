#!/bin/bash

# Setup script for lmgame_train_reorg
# Assumes you're already in the lmgame_train conda environment

set -e  # Exit on error

# Setup logging
LOG_FILE="setup_log.txt"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check if in conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_error "Not in a conda environment. Please activate lmgame_train environment first:"
        print_error "conda activate lmgame_train"
        exit 1
    fi
    
    print_success "Using conda environment: $CONDA_DEFAULT_ENV"
    
    # Check Python version
    if command_exists python; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        print_success "Python $PYTHON_VERSION detected"
    else
        print_error "Python not found"
        exit 1
    fi
    
    # Check git
    if ! command_exists git; then
        print_error "git is required but not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Initialize git submodules
setup_submodules() {
    print_step "Setting up git submodules..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository"
        exit 1
    fi
    
    # Initialize and update submodules
    print_step "Initializing git submodules..."
    git submodule init
    
    print_step "Updating git submodules..."
    git submodule update --recursive
    
    # Verify verl directory
    if [ -d "verl" ] && [ "$(ls -A verl)" ]; then
        print_success "verl submodule successfully downloaded"
    else
        print_error "Failed to download verl submodule"
        exit 1
    fi
}

# Install verl in editable mode
install_verl() {
    print_step "Installing verl framework..."
    
    if [ ! -d "verl" ]; then
        print_error "verl directory not found. Run git submodule update first."
        exit 1
    fi
    
    cd verl
    pip install -e . --no-dependencies
    cd ..
    
    print_success "verl installed in editable mode"
}

# Install torch first (required for flash-attn)
install_torch() {
    print_step "Installing torch 2.7.0..."
    
    # Check if torch is already installed
    if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.7.0"; then
        print_success "torch 2.7.0 already installed"
        return
    fi
    
    # Check for CUDA support
    if command_exists nvidia-smi; then
        print_step "CUDA GPU detected, installing torch with CUDA support..."
        pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    else
        print_step "No CUDA GPU detected, installing CPU-only torch..."
        pip install torch==2.7.0
    fi
    
    # Verify installation
    if python -c "import torch; print(f'torch {torch.__version__} installed successfully')" 2>/dev/null; then
        print_success "torch 2.7.0 installed successfully"
    else
        print_error "Failed to install torch"
        exit 1
    fi
}

# Install flash-attn with proper torch dependency
install_flash_attn() {
    print_step "Installing flash-attn..."
    
    # Check if flash-attn is already installed
    if python -c "import flash_attn" 2>/dev/null; then
        print_success "flash-attn already installed"
        return
    fi
    
    # Only install if CUDA is available
    if command_exists nvidia-smi; then
        print_step "Installing flash-attn==2.8.0.post2 (this may take several minutes)..."
        pip install flash-attn==2.8.0.post2 --no-build-isolation --no-cache-dir
        
        # Verify installation
        if python -c "import flash_attn" 2>/dev/null; then
            print_success "flash-attn installed successfully"
        else
            print_warning "flash-attn installation may have failed, but continuing..."
        fi
    else
        print_warning "No CUDA GPU detected, skipping flash-attn installation"
    fi
}

# Install remaining requirements
install_requirements() {
    print_step "Installing comprehensive requirements..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Create temporary requirements without torch and flash-attn (already installed)
    TEMP_REQ=$(mktemp)
    grep -v "^torch==" requirements.txt | \
    grep -v "^flash-attn==" | \
    grep -v "^#" | \
    grep -v "^$" > "$TEMP_REQ"
    
    if [ -s "$TEMP_REQ" ]; then
        print_step "Installing remaining dependencies..."
        pip install -r "$TEMP_REQ"
    fi
    
    # Clean up
    rm -f "$TEMP_REQ"
    
    print_success "All requirements installed successfully"
}

# Install this package in editable mode
install_package() {
    print_step "Installing lmgame_train package in editable mode..."
    pip install -e .
    print_success "lmgame_train package installed"
}

# Verify installation
verify_installation() {
    print_step "Verifying installation..."
    
    # Critical packages to test
    CRITICAL_PACKAGES=("torch" "transformers" "ray" "verl" "accelerate" "datasets" "wandb")
    
    for package in "${CRITICAL_PACKAGES[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            print_success "$package ✓"
        else
            print_warning "$package ✗"
        fi
    done
    
    # Test flash-attn separately (optional)
    if python -c "import flash_attn" 2>/dev/null; then
        print_success "flash_attn ✓"
    else
        print_warning "flash_attn ✗ (optional, only needed for CUDA)"
    fi
}

verify_submodule() {
    local path="$1"
    if [[ -d "$path" && $(ls -A "$path") ]]; then
        print_success "$path submodule downloaded"
    else
        print_error "Missing or empty submodule: $path"
        exit 1
    fi
}


# Check and setup authentication
setup_authentication() {
    print_step "Setting up authentication..."
    
    # Check for huggingface-cli
    if ! command -v huggingface-cli &> /dev/null; then
        print_warning "huggingface-cli not found. Installing..."
        pip install huggingface_hub[cli]
    fi
    
    # Check for wandb
    if ! command -v wandb &> /dev/null; then
        print_warning "wandb not found. Installing..."
        pip install wandb
    fi
    
    # Hugging Face login
    print_step "Checking Hugging Face authentication..."
    if [ -n "$HF_TOKEN" ]; then
        print_success "HF_TOKEN found in environment"
        print_step "Logging into Hugging Face..."
        if huggingface-cli login --token "$HF_TOKEN" >/dev/null 2>&1; then
            print_success "Hugging Face login successful"
        else
            print_warning "Hugging Face login failed, but continuing..."
        fi
    else
        print_warning "HF_TOKEN not found in environment"
        echo "   To set HF_TOKEN: export HF_TOKEN=your_hf_token"
    fi
    
    # Weights & Biases login
    print_step "Checking Weights & Biases authentication..."
    if [ -n "$WANDB_API_KEY" ]; then
        print_success "WANDB_API_KEY found in environment"
        print_step "Logging into Weights & Biases..."
        if wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1; then
            print_success "Weights & Biases login successful"
        else
            print_warning "Weights & Biases login failed, but continuing..."
        fi
    else
        print_warning "WANDB_API_KEY not found in environment"
        echo "   To set WANDB_API_KEY: export WANDB_API_KEY=your_wandb_key"
    fi
    
    # Set WANDB team/organization
    if [ -n "$WANDB_ENTITY" ]; then
        print_success "WANDB_ENTITY found: $WANDB_ENTITY"
    else
        print_warning "WANDB_ENTITY not found in environment"
        echo "   To set WANDB_ENTITY: export WANDB_ENTITY=your_wandb_org_name"
    fi
    
    echo ""
    echo "💡 For better security, consider setting environment variables:"
    echo "   export HF_TOKEN=your_hf_token"
    echo "   export WANDB_API_KEY=your_wandb_key"
    echo "   export WANDB_ENTITY=your_wandb_org_name"
    echo ""
}

# Main installation function
main() {
    echo "=========================================="
    echo "lmgame_train_reorg Setup Script"
    echo "Started at: $(date)"
    echo "=========================================="
    
    check_prerequisites
    setup_submodules
    install_verl
    install_torch
    install_flash_attn
    install_requirements
    install_package
    verify_installation
    setup_authentication
    verify_submodule verl
    verify_submodule external/webshop-minimal
    
    echo "=========================================="
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo "Setup completed at: $(date)"
    echo "Full log saved to: $LOG_FILE"
    echo "=========================================="
    echo ""
    echo "You can now run your training scripts:"
    echo "  python train.py"
    echo ""
    echo "Current environment: $CONDA_DEFAULT_ENV"
}

# Run main function
main