#!/usr/bin/env python3
"""
Post-installation script to set up submodules for LMGameRL.
This runs after pip install -e . to configure submodules.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, cwd=None, check=True):
    """Run a command with proper error handling."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            check=check
        )
        if result.returncode == 0:
            print(f"✅ {description}")
            return True
        else:
            print(f"⚠️  {description} (with warnings)")
            if result.stderr:
                print(f"   {result.stderr.strip()}")
            return not check
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"   Error: {e.stderr.strip()}")
        return False

def setup_submodules():
    """Initialize and configure submodules."""
    print("📦 Setting up LMGameRL submodules...")
    
    # Check if we're in a git repository
    if not run_command("git rev-parse --git-dir", "Checking git repository", check=False):
        print("⚠️  Not in a git repository - submodules will need manual setup")
        return False
    
    # Initialize submodules
    if not run_command("git submodule init", "Initializing submodules"):
        return False
    
    # Update submodules
    if not run_command("git submodule update --recursive", "Updating submodules"):
        return False
    
    # Verify submodules
    required_submodules = ["verl", "external/webshop-minimal"]
    for submodule in required_submodules:
        path = Path(submodule)
        if path.exists() and any(path.iterdir()):
            print(f"✅ Submodule {submodule} ready")
        else:
            print(f"❌ Submodule {submodule} not found or empty")
            return False
    
    return True

def install_verl():
    """Install verl in editable mode without dependencies."""
    print("\n🔧 Installing verl framework...")
    
    verl_path = Path("verl")
    if not verl_path.exists():
        print("❌ verl submodule not found")
        return False
    
    # Install verl without dependencies to avoid conflicts
    return run_command(
        "pip install -e . --no-dependencies", 
        "Installing verl (editable, no-deps)",
        cwd=verl_path
    )

def install_webshop():
    """Install webshop-minimal submodule."""
    print("\n🛒 Installing webshop-minimal...")
    
    webshop_path = Path("external/webshop-minimal")
    if not webshop_path.exists():
        print("⚠️  WebShop submodule not found")
        return False
    
    # Install webshop in editable mode
    success = run_command(
        "pip install -e .", 
        "Installing webshop-minimal (editable)",
        cwd=webshop_path,
        check=False
    )
    
    if success:
        # Download spacy models
        run_command("python -m spacy download en_core_web_sm", "Downloading spacy small model", check=False)
        run_command("python -m spacy download en_core_web_lg", "Downloading spacy large model", check=False)
    
    return success

def main():
    """Main installation process."""
    print("🚀 LMGameRL Submodule Setup")
    print("=" * 40)
    
    if not setup_submodules():
        print("❌ Submodule setup failed")
        sys.exit(1)
    
    if not install_verl():
        print("❌ VERL installation failed")
        sys.exit(1)
    
    # WebShop installation is optional
    install_webshop()
    
    print("\n🎉 Submodule setup completed!")
    print("\nNext steps:")
    print("  • Load datasets: python scripts/load_dataset.sh --all")
    print("  • Run training: lmgamerl-train")

if __name__ == "__main__":
    main()