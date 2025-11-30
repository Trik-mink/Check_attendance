# 🚀 How to Run the Face Recognition Attendance System

## ✅ Clean Environment Setup Complete!

You now have a **fresh, clean environment** (`faceenv_cam`) with no dependency conflicts.

---

## 🎯 Quick Start (Easiest Way)

### Option 1: Use the Run Script (Recommended)

```bash
cd /Users/tristan/Check_attendance
./run_app.sh
```

This script automatically:
- Activates the correct conda environment
- Verifies Python version
- Launches Streamlit

---

### Option 2: Manual Command

```bash
# Activate the environment
conda activate faceenv_cam

# Run the app
cd /Users/tristan/Check_attendance
streamlit run streamlit_app.py
```

---

## 📸 Using the App

1. **Open your browser** to `http://localhost:8501` (should open automatically)

2. **Click "Take a photo for check-in"** - This uses your browser's camera (no OpenCV!)

3. **Allow camera access** when prompted by your browser

4. **Click the camera button** in the Streamlit widget to capture

5. **See results instantly:**
   - ✅ Employee name and similarity score
   - 🏆 Top 4 matching faces
   - 🌌 3D visualization of face embeddings
   - 📊 Similarity table

---

## 🔧 What Changed?

### ✅ Improvements:

1. **Clean Environment** - No more dependency conflicts
   - Python 3.11
   - NumPy 1.26.4 (not 2.x)
   - No OpenCV
   - No scikit-learn

2. **Browser Camera** - Uses `st.camera_input` instead of OpenCV
   - No segmentation faults
   - Works on all platforms
   - More secure (browser-controlled)

3. **Pure NumPy PCA** - No sklearn dependency
   - SVD-based dimensionality reduction
   - Fast and reliable

4. **Fixed FAISS Caching** - Uses `st.session_state`
   - No C++ object pickling
   - No crashes

---

## 🎨 Features

- **Employee Dashboard** - See all employees and check-in status
- **Live Camera** - Browser-based camera capture
- **Face Recognition** - FaceNet embeddings + FAISS search
- **3D Visualization** - Interactive Plotly 3D scatter plot
- **Match Display** - Top matches with similarity scores
- **Grid Layout** - Clean, professional UI

---

## 🛑 To Stop the Server

Press `Ctrl+C` in the terminal

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'plotly'"
**Cause:** Running in wrong environment

**Fix:** Use the `run_app.sh` script or make sure you activated `faceenv_cam`:
```bash
conda activate faceenv_cam
```

### Camera not working
**Fix:** 
- Make sure you allowed camera access in your browser
- Try Chrome or Firefox (Safari sometimes has issues)
- Check that no other app is using the camera

### "FAISS index not found"
**Fix:** Build the index first:
```bash
conda activate faceenv_cam
python facenet_app.py
```

---

## 📦 Installed Packages (faceenv_cam)

```
✅ streamlit 1.51.0
✅ numpy 1.26.4
✅ torch 2.2.0
✅ torchvision 0.17.0
✅ facenet-pytorch 2.6.0
✅ faiss-cpu 1.13.0
✅ pillow 10.2.0
✅ plotly 6.5.0
✅ pandas 2.3.3
```

**NOT installed (to avoid conflicts):**
- ❌ opencv-python
- ❌ scikit-learn
- ❌ matplotlib
- ❌ numpy 2.x

---

## 🎓 Technical Details

### Why `faceenv_cam` instead of the old environment?

The old environment had:
- NumPy 2.x conflicting with older packages
- OpenCV causing ABI conflicts
- sklearn causing segmentation faults

The new environment:
- Uses stable NumPy 1.26.4
- No OpenCV (uses Streamlit's camera)
- No sklearn (uses pure NumPy)
- Clean slate = no conflicts

### How does `st.camera_input` work?

- Uses browser's native `getUserMedia()` API
- More secure (browser controls permissions)
- Cross-platform (works on Mac, Windows, Linux)
- No native dependencies

---

## 📚 File Structure

```
Check_attendance/
├── streamlit_app.py      # Main web app (UPDATED - no OpenCV/sklearn)
├── facenet_app.py        # Index builder
├── pixel_app.py          # Pixel-based matching
├── run_app.sh           # Easy run script (NEW!)
├── Dataset/              # Employee images
├── facenet_features.index  # FAISS index
└── facenet_label_map.npy   # Employee labels
```

---

## 🎉 You're All Set!

Run the app with:
```bash
./run_app.sh
```

Or manually:
```bash
conda activate faceenv_cam
streamlit run streamlit_app.py
```

**Enjoy your face recognition system!** 🎭✨

