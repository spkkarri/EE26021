#!/bin/bash

echo "======================================"
echo "   SMART GRID DATA DOWNLOAD SCRIPT    "
echo "======================================"

# Create directory if not exists
mkdir -p datasets
cd datasets

echo "Downloading Smart Grid Stability Dataset..."
git clone https://github.com/srinidhis05/Smart-Grid-Stability-Dataset.git

echo "Downloading Household Power Consumption Dataset..."
wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/household_power_consumption.csv

echo "Downloading Energy Dataset..."
git clone https://github.com/rob-med/energydata_complete.git

echo "======================================"
echo "   DOWNLOAD COMPLETED SUCCESSFULLY    "
echo "======================================"
