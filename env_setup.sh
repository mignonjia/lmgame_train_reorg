#!/bin/bash

# Environment Setup Script for LMGame Training
set -e  # Exit on any error

echo "🚀 Setting up environment for LMGame training..."

# Check dependencies
echo "🔍 Checking dependencies..."

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli not found. Installing..."
    pip install huggingface_hub[cli]
fi

# Check for wandb
if ! command -v wandb &> /dev/null; then
    echo "❌ wandb not found. Installing..."
    pip install wandb
fi

echo "✅ Dependencies checked"

# Hugging Face login
echo "📦 Logging into Hugging Face..."
if [ -n "$HF_TOKEN" ]; then
    echo "Using HF_TOKEN from environment"
    huggingface-cli login --token "$HF_TOKEN"
else
    # Fallback to hardcoded token (not recommended for production)
    echo "⚠️  Using hardcoded token - consider setting HF_TOKEN environment variable"
huggingface-cli login --token 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
fi

# Weights & Biases login
echo "📊 Logging into Weights & Biases..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "Using WANDB_API_KEY from environment"
    wandb login --relogin "$WANDB_API_KEY"
else
    # Fallback to hardcoded key
    echo "⚠️  Using hardcoded API key - consider setting WANDB_API_KEY environment variable"
    export WANDB_API_KEY=5c18e0e1920e548d7cd21774c89c6e9a28facc65
    wandb login --relogin "$WANDB_API_KEY"
fi

# Set WANDB team/organization
echo "🏢 Setting WANDB team/organization..."
if [ -z "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY=yuxuan_zhang13-uc-san-diego
    echo "Set WANDB_ENTITY to: $WANDB_ENTITY"
else
    echo "Using existing WANDB_ENTITY: $WANDB_ENTITY"
fi

echo "✅ Environment setup complete!"
echo ""
echo "💡 For better security, consider setting environment variables:"
echo "   export HF_TOKEN=your_hf_token"
echo "   export WANDB_API_KEY=your_wandb_key"
echo "   export WANDB_ENTITY=your_wandb_org_name"