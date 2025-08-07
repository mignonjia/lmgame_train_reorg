#!/bin/bash
# scripts/load_dataset.sh - Main dataset loading script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Environment variables
LOAD_BIRD_DATASET=${LOAD_BIRD_DATASET:-0}
LOAD_WEBSHOP_DATASET=${LOAD_WEBSHOP_DATASET:-0}
LOAD_ALL_DATASETS=${LOAD_ALL_DATASETS:-0}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bird)
            LOAD_BIRD_DATASET=1
            shift
            ;;
        --webshop)
            LOAD_WEBSHOP_DATASET=1
            shift
            ;;
        --all)
            LOAD_ALL_DATASETS=1
            shift
            ;;
        -h|--help)
            echo "Dataset Loading Script for lmgamerl"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --bird      Load Bird dataset"
            echo "  --webshop   Load WebShop dataset"
            echo "  --all       Load all datasets"
            echo "  -h, --help  Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  LOAD_BIRD_DATASET=1     Load Bird dataset"
            echo "  LOAD_WEBSHOP_DATASET=1  Load WebShop dataset"
            echo "  LOAD_ALL_DATASETS=1     Load all datasets"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set flags for --all
if [[ "$LOAD_ALL_DATASETS" == 1 ]]; then
    LOAD_BIRD_DATASET=1
    LOAD_WEBSHOP_DATASET=1
fi

echo "=========================================="
echo "lmgamerl Dataset Loading Script"
echo "Started at: $(date)"
echo "=========================================="

# Load Bird dataset using Python script
if [[ "$LOAD_BIRD_DATASET" == 1 ]]; then
    print_step "Loading Bird dataset..."
    if python scripts/load_dataset.py --bird; then
        print_success "Bird dataset loaded successfully"
    else
        print_error "Failed to load Bird dataset"
        exit 1
    fi
fi

# Load WebShop dataset directly in shell
if [[ "$LOAD_WEBSHOP_DATASET" == 1 ]]; then
    print_step "Loading WebShop dataset..."
    
    # Check if webshop submodule exists
    if [[ ! -d "external/webshop-minimal" ]]; then
        print_error "WebShop submodule not found. Please run git submodule update --init first."
        exit 1
    fi
    
    # Create data directory
    WEBSHOP_DATA_DIR="external/webshop-minimal/webshop_minimal/data/full"
    mkdir -p "$WEBSHOP_DATA_DIR"
    
    # Download WebShop datasets
    cd "$WEBSHOP_DATA_DIR"
    
    if ! command -v gdown &> /dev/null; then
        print_error "gdown not found. Please install it: pip install gdown"
        exit 1
    fi
    
    print_step "Downloading items_shuffle..."
    if gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB -O items_shuffle.json; then
        print_success "items_shuffle downloaded"
    else
        print_error "Failed to download items_shuffle"
        exit 1
    fi
    
    print_step "Downloading items_ins_v2..."
    if gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi -O items_ins_v2.json; then
        print_success "items_ins_v2 downloaded"
    else
        print_error "Failed to download items_ins_v2"
        exit 1
    fi
    
    cd - > /dev/null
    print_success "WebShop dataset loaded successfully"
fi

# Show summary
if [[ "$LOAD_BIRD_DATASET" == 0 && "$LOAD_WEBSHOP_DATASET" == 0 ]]; then
    print_warning "No datasets specified to load."
    echo ""
    echo "Usage examples:"
    echo "  $0 --bird          # Load Bird dataset only"
    echo "  $0 --webshop       # Load WebShop dataset only" 
    echo "  $0 --all           # Load all datasets"
    echo ""
    echo "Or use environment variables:"
    echo "  LOAD_BIRD_DATASET=1 $0"
    echo "  LOAD_WEBSHOP_DATASET=1 $0"
fi

echo "=========================================="
echo -e "${GREEN}Dataset loading completed!${NC}"
echo "Completed at: $(date)"
echo "=========================================="