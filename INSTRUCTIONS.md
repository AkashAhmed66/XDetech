# X-Ray Analysis Application

This application uses a GhostNet deep learning model to analyze chest X-rays for three conditions:
- Normal
- Tuberculosis
- Corona Virus

## Quick Start

The easiest way to run the application is using the helper script:

```bash
python run.py
```

This script will:
1. Check if all required dependencies are installed
2. Install any missing dependencies
3. Start the application

Once started, access the application at: http://127.0.0.1:5000

## Manual Setup

If you prefer to set up manually:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

## Requirements

- Python 3.8 or higher
- PyTorch
- Flask
- OpenCV (for visualization)
- Other dependencies listed in requirements.txt

## Features

- **Model**: Uses GhostNet for lightweight, efficient image classification
- **GradCAM Visualization**: Custom implementation that highlights the areas of the X-ray image that influence the model's decision
- **Interactive UI**: Clean, responsive interface with detailed results

## Model Files

The application looks for model weights in either:
- `model.pth` 
- `ghostnet.pth`

Make sure one of these files is present in the root directory.

## Troubleshooting

### GradCAM Visualization Issues

If the GradCAM visualization shows only a blue overlay or error message:

1. The issue may be related to the target layer selection. The application automatically tries to find the most appropriate convolutional layer from the GhostNet model for visualization. 

2. Check the application logs for specific error messages about gradient capture or target layers.

3. Potential fixes:
   - Ensure the model is in evaluation mode with `model.eval()`
   - Check that image normalization is consistent
   - Try with different X-ray images

### Image Display Problems

If the original image doesn't display correctly:

1. Check that the uploaded image is saved correctly in the static/uploads folder
2. Verify the file path is correctly passed to the template
3. Ensure the image format is supported (JPG, PNG)

### Dependencies Installation Problems

For Windows users experiencing issues with package installation:

1. Try updating pip:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Install packages individually:
   ```bash
   pip install torch torchvision
   pip install timm
   pip install opencv-python
   pip install -r requirements.txt
   ```

3. Use conda instead of pip if you're using Anaconda:
   ```bash
   conda install pytorch torchvision -c pytorch
   conda install -c conda-forge opencv
   pip install timm
   ```

## Contact

If you encounter persistent issues, please report them with:
1. Full error message/traceback
2. Python and OS version
3. Steps to reproduce the issue 