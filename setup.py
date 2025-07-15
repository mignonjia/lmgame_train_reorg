from setuptools import setup, find_packages
import os

# Read the contents of README file
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt, excluding comments and empty lines"""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and section headers
                if line and not line.startswith("#") and not line.startswith("="):
                    requirements.append(line)
    return requirements

setup(
    name="lmgame_train",
    version="0.1.0",
    description="Multi-Turn PPO Training System with AgentTrainer",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="lmgame_train Team",
    author_email="",
    url="",
    
    # Package configuration
    packages=find_packages(),
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Package data
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml", "*.json"],
    },
    include_package_data=True,
    
    # Entry points
    entry_points={
        "console_scripts": [
            "lmgame-train=train:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords="reinforcement learning, PPO, multi-turn, training, AI, machine learning",
    
    # Project URLs
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
    
    # Additional metadata
    zip_safe=False,
    
    # Development dependencies (optional)
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
) 