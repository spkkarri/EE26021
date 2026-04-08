# PowerShell script to download dataset from Google Drive
pip install gdown

# Replace this link with your dataset folder or file link
$folderUrl = "https://drive.google.com/drive/folders/1MbgZrASiTTbc7bM3Yi368yPJ6obH7a6o?usp=sharing"

# Set destination directory
$destinationDir = ".\11-114\data"

# Create directory if it doesn't exist
if (-not (Test-Path -Path $destinationDir)) {
    New-Item -ItemType Directory -Path $destinationDir | Out-Null
}

# Download the dataset
Write-Host "Downloading dataset from Google Drive..."
python -m gdown --folder $folderUrl -O $destinationDir
Write-Host "`nDownload completed!"
