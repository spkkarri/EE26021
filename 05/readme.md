# Pothole Detection
This project implements pothole detection using computer vision techniques. 
It processes road images to identify potholes, draws bounding boxes around them, 
and estimates their dimensions using image processing methods such as grayscale conversion, 
Gaussian blur, thresholding, and contour detection.

## Team Members
Srihitha Tirnati - 524183

Kshirja Challa - 524143

P Bhavya - 524163

## Project Explanation
Watch our project explanation:
[![Link]](https://drive.google.com/file/d/1GWo6X1gM-GqVM1VRB5ypNb03OOUdvPeO/view?usp=sharing)

Due to GitHub file size limitations, the dataset is not included in this repository.

You can download the dataset from the following link:

👉 https://drive.google.com/drive/folders/1v5DMse4US5p5l_Ep4dLlEWPAnJRJfJOu?usp=sharing

## Setup Instructions


### 1. Install Requirements
Run the following in Google Colab:

pip install ultralytics opencv-python matplotlib numpy

### 2. Upload Dataset

Upload your dataset zip file using:

from google.colab import files
files.upload()

Then extract:

import zipfile

with zipfile.ZipFile("dataset.zip", 'r') as zip_ref:

    zip_ref.extractall("/content/dataset")

### 3. Run the Project

Execute the main code in Colab:

- Load image
- Apply preprocessing (grayscale, blur)
- Apply thresholding
- Detect contours
- Draw bounding boxes

The output will show detected potholes with dimensions.


### 4. Results and Outputs

- Potholes detected using bounding boxes
- Width and height displayed on image
- Output visualized using matplotlib

## Project Structure

Code/

  pothole_detection.ipynb
  README.md
  

assets/
  PPT.pptx
  Report.pdf
  

data/
  dataset.zip
  

## Model Performance

- Successfully detects potholes using contour detection
- Works well on clear road images
- Accuracy depends on lighting and threshold values
## System Requirements

- Python 3.x
- Google Colab
- OpenCV
- NumPy
- Matplotlib
