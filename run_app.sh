#!/bin/bash
# Script to run the Streamlit app in the correct conda environment

echo "🚀 Starting Face Recognition Attendance System..."
echo "📦 Activating faceenv_cam environment..."

# Activate the conda environment
source /Users/tristan/miniconda3/bin/activate faceenv_cam

# Verify we're in the right environment
echo "✅ Python: $(which python)"
echo "✅ Python version: $(python --version)"

# Run Streamlit
echo "🎬 Launching Streamlit app..."
cd /Users/tristan/Check_attendance
streamlit run streamlit_app.py

