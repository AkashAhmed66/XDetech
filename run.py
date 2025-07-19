#!/usr/bin/env python
"""
Helper script to check dependencies and run the X-ray analysis application.
"""
import subprocess
import sys
import importlib
import os

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'flask',
        'torch',
        'torchvision',
        'timm',
        'pytorch_grad_cam',
        'matplotlib',
        'numpy',
        'Pillow',
        'opencv-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is NOT installed")
    
    return missing_packages

def install_missing_packages(packages):
    """Install missing packages using pip."""
    if not packages:
        return True
    
    print("\nInstalling missing packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install them manually using:")
        print(f"pip install {' '.join(packages)}")
        return False

def run_app():
    """Run the Flask application."""
    try:
        from app import app
        app.run(debug=True)
    except Exception as e:
        print(f"❌ Error running application: {str(e)}")
        return False
    return True

if __name__ == "__main__":
    print("Checking dependencies for X-ray Analysis Application...")
    missing = check_dependencies()
    
    if missing:
        print("\nSome dependencies are missing. Installing them...")
        if not install_missing_packages(missing):
            sys.exit(1)
    
    print("\nStarting X-ray Analysis Application...")
    if not run_app():
        sys.exit(1) 