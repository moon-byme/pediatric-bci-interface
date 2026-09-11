$ModelDirectory = Join-Path $PSScriptRoot "models"
$OutputFile = Join-Path $ModelDirectory "pose_landmarker_full.task"
$ModelUrl = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

New-Item -ItemType Directory -Force -Path $ModelDirectory | Out-Null

Write-Host "Downloading the official MediaPipe Pose Landmarker Full model..."
Invoke-WebRequest -Uri $ModelUrl -OutFile $OutputFile

Write-Host "Saved to: $OutputFile"
Write-Host "Run: python main.py"
