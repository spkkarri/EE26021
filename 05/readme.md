# Pothole Detection
This project implements pothole detection using computer vision techniques. It processes road images to identify potholes, draws bounding boxes around them, and estimates their dimensions using image processing methods such as grayscale conversion, Gaussian blur, thresholding, and contour detection.


## Team Members
Kshirja Challa - 524143

Srihitha Tirnati - 524183

P Bhavya - 524163

## Project Explanation
Watch our project explanation:
[![Link]](https://drive.google.com/file/d/1GWo6X1gM-GqVM1VRB5ypNb03OOUdvPeO/view?usp=sharing)

## Setup Instructions


### 1. Install Requirements
Open terminal in VS Code and run:

pip install ultralytics opencv-python matplotlib numpy

### 2. Upload Dataset


### 3. Run the Project

Run the projectt using vs code

Run the script:
python train.py

The output will show detected potholes with dimensions.


### 4. Results and Outputs

- Potholes detected using bounding boxes
- Width and height displayed on image
- Output visualized using matplotlib

## Project Structure
05/
 ├── code/
 │     └── train.py
 ├── asset/
 │     ├── PPT.pptx
 │     └── Report.pdf
 ├── data/
 │     └── dataset_link.txt
 └── readme.md
  

## Model Performance

- Successfully detects potholes using contour detection
- Works well on clear road images
- Accuracy depends on lighting and threshold values
## System Requirements

- Python 3.x
- Vs code
- OpenCV
- NumPy
- Matplotlib

Dataset:
Download dataset from the link provided in the data folder and extract it.
