# Face Recognition Attendance System

## 📖 Project Overview

This project implements a simple **employee image search** and **face similarity system** designed to find the most similar employee images in a dataset given a query image. The system explores two fundamentally different approaches to image similarity:

**Approach 1 (V3)** uses **raw pixel-based similarity**: each image is flattened into a 270,000-dimensional vector (300×300×3 RGB values), and similarity is computed using Euclidean distance (L2) via FAISS. This approach works well when the exact same image exists in the dataset but struggles with new photos of the same person due to variations in lighting, pose, or background.

**Approach 2 (V4/V5)** uses **deep learning-based embeddings**: a pre-trained FaceNet model (InceptionResnetV1 trained on VGGFace2) extracts a compact 512-dimensional face embedding for each image. These embeddings capture high-level facial features rather than raw pixels, making them robust to lighting changes, slight pose variations, and other transformations. Similarity is measured using inner product (cosine similarity) via FAISS, enabling more accurate face recognition even on new images of the same person.

The main goal is to demonstrate the difference between traditional pixel-based methods and modern deep learning approaches for face recognition tasks.

---

## 🛠️ Requirements & Environment Setup

This project was tested in a **conda environment** with **Python 3.11** (or 3.13). The key dependencies include:

- `numpy` — numerical operations
- `pandas` — data manipulation (optional, for organizing paths/labels)
- `Pillow` — image loading and preprocessing
- `matplotlib` — visualization
- `faiss-cpu` — fast vector similarity search
- `torch` — PyTorch deep learning framework
- `torchvision` — image transformations
- `facenet-pytorch` — pre-trained FaceNet model
- `streamlit` — web UI framework (optional, for the web interface)

### Setup Instructions

#### Option 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n faceenv python=3.11 -y
conda activate faceenv

# Install basic dependencies
pip install numpy pandas matplotlib Pillow faiss-cpu

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install FaceNet
pip install facenet-pytorch
```

#### Option 2: Using Miniconda (if already installed)

If you already have miniconda installed, simply activate your environment and install the dependencies:

```bash
python -m pip install numpy pandas matplotlib Pillow faiss-cpu torch torchvision facenet-pytorch
```

### VS Code Setup

To use this environment in **VS Code**:

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type **"Python: Select Interpreter"**
3. Choose the `faceenv` conda environment

---

## 📁 Dataset Structure

The `Dataset/` directory contains employee avatar images with a specific naming convention:

```
Dataset/
├── Avatar_Anh_Khoi.jpg
├── Avatar_Ngo_Ngoc.jpg
├── Avatar_Thanh_Tam.jpg
├── Avatar_Minh_Chau.jpg
├── Avatar_Aaron_Eckhart.jpg
└── ...
```

### Naming Convention

- **All filenames start with `Avatar_`**
- The code automatically extracts the employee name by:
  1. Removing the file extension (`.jpg`, `.jpeg`, `.png`)
  2. Stripping the `"Avatar_"` prefix (first 7 characters)
- The resulting string becomes the **employee label**
  - Example: `Avatar_Ngo_Ngoc.jpg` → label = `Ngo_Ngoc`

### Supported Formats

The code accepts images in the following formats: `.jpg`, `.jpeg`, `.png` (case-insensitive).

---

## 🎨 Approach 1 — Pixel-Based Similarity (V3)

### Overview

The `pixel_app.py` script implements a **baseline image similarity system** using raw pixel values. This approach treats images as high-dimensional vectors without any semantic understanding of the content.

### How It Works

1. **Load images** from `Dataset/`
2. **Resize** each image to 300×300 pixels
3. **Convert to RGB** (handles grayscale images by duplicating channels)
4. **Normalize** pixel values to the range [0, 1]
5. **Flatten** the image into a **270,000-dimensional vector** (300 × 300 × 3)
6. **Build a FAISS index** using `IndexFlatL2(270000)` for Euclidean (L2) distance
7. **Save the index** to disk:
   - `employee_images.index` — FAISS index file
   - `label_map.npy` — numpy array of employee labels

### API Functions

- **`image_to_vector(image_path)`** — Converts an image to a 270,000-dim vector
- **`search_similar_images_pixel(query_image_path, k=5)`** — Returns top-k similar employees with L2 distances
- **`display_query_and_top_matches_pixel(query_image_path)`** — Displays the query image and top-5 matches

### Running the Script

```bash
python pixel_app.py
```

### Output

The script will:

1. Display the **query image**
2. Show the **top-5 most similar images** with their **L2 distance values**
   - **Lower distance** = more similar
   - Distance of 0 = exact match

### Example Output

```
✓ Loaded 20 employee images
Building pixel-based FAISS index...
✓ Saved pixel-based index with 20 employees

==================================================
V3: PIXEL-BASED FACE RECOGNITION
==================================================
```

Two matplotlib windows will appear:

- Window 1: Your query image
- Window 2: Top 5 matches with distance scores

### Limitations

⚠️ **This approach has significant limitations:**

- ✅ Works well if the **exact same image** exists in the dataset
- ❌ Performs poorly on **new images** of the same person
- ❌ Sensitive to:
  - Lighting changes
  - Background differences
  - Slight pose variations
  - Image compression artifacts
- ❌ High dimensionality (270,000) makes it computationally expensive

**Conclusion:** Pixel-based similarity is not suitable for real-world face recognition because it lacks semantic understanding of facial features.

---

## 🧠 Approach 2 — FaceNet Embedding Similarity (V4/V5)

### Overview

The `facenet_app.py` script implements a **modern face recognition system** using deep learning embeddings. Instead of raw pixels, it uses a pre-trained neural network to extract meaningful facial features.

### How It Works

1. **Scan** `Dataset/` and extract labels (same as V3)
2. **Load FaceNet model:**
   ```python
   InceptionResnetV1(pretrained='vggface2').eval()
   ```
   - Pre-trained on the VGGFace2 dataset (millions of face images)
   - Converts faces to 512-dimensional embeddings
3. **Preprocess images** with torchvision transforms:
   - Resize to (300, 300)
   - Convert to PyTorch tensor
   - Normalize to mean=0.5, std=0.5 (values in [-1, 1])
4. **Compute 512-dimensional embeddings** for each image
5. **L2-normalize embeddings** so that inner product ≈ cosine similarity
6. **Build a FAISS index** using `IndexFlatIP(512)` for inner-product similarity
7. **Save the index** to disk:
   - `facenet_features.index` — FAISS index file
   - `facenet_label_map.npy` — numpy array of employee labels

### API Functions

- **`image_to_feature(image_path)`** — Converts an image to a normalized 512-dim embedding
- **`search_similar_images_facenet(query_image_path, k=5)`** — Returns top-k similar employees with similarity scores
- **`display_query_and_top_matches_facenet(query_image_path, k=5)`** — Displays the query image and top-k matches

### Running the Script

```bash
python facenet_app.py
```

### Output

The script will:

1. Display the **query image**
2. Show the **top-5 most similar employee images** with **similarity scores**
   - **Higher score** = more similar
   - Score ≈ 1.0 = very similar (cosine similarity)

### Example Output

```
Scanning dataset...
✓ Loaded 20 employee images
Loading FaceNet model...
Downloading: "https://github.com/timesler/facenet-pytorch/releases/download/..."
✓ FaceNet model loaded OK
Importing faiss and building index...
✓ Added 20 vectors to FAISS index
Running FaceNet search for: Dataset/Avatar_Ngo_Ngoc.jpg
```

Two matplotlib windows will appear:

- Window 1: Your query image
- Window 2: Top 5 matches with similarity scores

### Why This Approach is Better

✅ **Advantages over pixel-based (V3):**

- **Semantic understanding:** 512-dim embeddings capture facial identity, not raw pixels
- **Robustness:** Works well even with:
  - Different lighting conditions
  - Slight pose variations
  - Different backgrounds
  - Image quality differences
- **Generalization:** Can recognize new photos of the same person
- **Efficiency:** 512 dimensions vs 270,000 dimensions
- **Real-world applicability:** This is how modern face recognition systems work (e.g., Face ID, Facebook photo tagging)

### How Face Embeddings Work

The FaceNet model learns to:

- Place images of the **same person close together** in 512-dimensional space
- Place images of **different people far apart**
- Ignore irrelevant factors (lighting, background, minor pose changes)

This is achieved through **triplet loss training**:

- Anchor: a face image
- Positive: another image of the same person
- Negative: an image of a different person
- Loss encourages: `distance(anchor, positive) < distance(anchor, negative)`

---

## 📊 Comparison: V3 vs V4/V5

| Feature                    | V3 (Pixel-Based)         | V4/V5 (FaceNet)           |
| -------------------------- | ------------------------ | ------------------------- |
| **Vector Dimension**       | 270,000                  | 512                       |
| **Distance Metric**        | L2 (Euclidean)           | Inner Product (Cosine)    |
| **Feature Type**           | Raw pixels               | Deep learning embeddings  |
| **Semantic Understanding** | ❌ None                  | ✅ Captures face identity |
| **Robustness**             | ❌ Poor                  | ✅ Excellent              |
| **Works on new images?**   | ❌ No (exact match only) | ✅ Yes (same person)      |
| **Computational Cost**     | High (270k dims)         | Low (512 dims)            |
| **Real-world Use**         | ❌ Not practical         | ✅ Industry standard      |
| **Training Required**      | None                     | Pre-trained model         |

### When to Use Each Approach

**Use V3 (Pixel-Based) when:**

- You only need exact duplicate detection
- You're doing image deduplication
- Educational purposes (understanding baselines)

**Use V4/V5 (FaceNet) when:**

- You need real face recognition
- You want to recognize people in new photos
- You're building a production system
- You need robustness to variations

---

## 🚀 How to Run the Project

### Quick Start

1. **Activate your environment:**

   ```bash
   conda activate faceenv
   # or if using miniconda directly
   python --version  # should be Python 3.11 or 3.13
   ```

2. **Run the pixel-based approach:**

   ```bash
   python pixel_app.py
   ```

3. **Run the FaceNet approach:**
   ```bash
   python facenet_app.py
   ```

### Customizing the Query Image

To test with a different employee image, edit the last line of either script:

**In `pixel_app.py`:**

```python
if __name__ == "__main__":
    display_query_and_top_matches_pixel("Dataset/Avatar_YOUR_IMAGE.jpg")
```

**In `facenet_app.py`:**

```python
if __name__ == "__main__":
    test_path = "Dataset/Avatar_YOUR_IMAGE.jpg"
    display_query_and_top_matches_facenet(test_path, k=5)
```

---

## 🌐 Web Interface (Streamlit UI)

**NEW!** A beautiful web interface is now available for easier interaction!

### Why Use the Streamlit UI?

✅ **No Code Editing** - Upload images directly through your browser

✅ **Interactive** - Adjust the number of matches with a slider

✅ **Visual** - Beautiful UI with image previews and color-coded similarity scores

✅ **User-Friendly** - Perfect for demos and non-technical users

### Running the Streamlit App

1. **First, build the FAISS index** (required, only once):

   ```bash
   python facenet_app.py
   ```

2. **Start the Streamlit server:**

   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** to `http://localhost:8501`

   The app should open automatically, but if not, visit the URL manually.

### Features

- 📤 **Drag-and-drop** image upload
- 🎚️ **Adjustable** number of matches (1-10)
- 🎨 **Color-coded** similarity scores:
  - 🟢 Very Similar (> 0.8)
  - 🟡 Similar (0.6 - 0.8)
  - 🔴 Less Similar (< 0.6)
- 📊 **Detailed results** with rankings
- 👥 **Sample images** from the database
- ⚡ **Fast loading** with cached model

### Screenshot Preview

The app displays:

- Your uploaded query image
- Top-K most similar employees
- Similarity scores and rankings
- Employee names and photos

### Stopping the Server

Press `Ctrl+C` in the terminal to stop the Streamlit server.

For detailed instructions, see **[STREAMLIT_GUIDE.md](STREAMLIT_GUIDE.md)**.

---

## 🔮 Future Work & Extensions

Here are some ideas to extend this project:

### 1. Face Detection & Cropping

- **Current:** Assumes images are already face-centered
- **Improvement:** Use MTCNN or other face detectors to:
  - Detect faces in images
  - Crop and align faces before embedding
  - Handle multiple faces per image

```python
from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=300, margin=0)
face_img = mtcnn(img)  # detect and crop face
```

### 2. Web Interface

Build a simple web app for easier interaction:

**Option A: Streamlit**

```python
import streamlit as st
uploaded_file = st.file_uploader("Upload a face image")
if uploaded_file:
    results = search_similar_images_facenet(uploaded_file, k=5)
    st.image(results)
```

**Option B: Flask/FastAPI**

- Create REST API endpoints
- Upload query image via POST request
- Return JSON with top matches

### 3. Evaluation Metrics

Add quantitative evaluation:

- Split dataset into training/test sets
- Compute **top-1 accuracy**: Is the correct person ranked #1?
- Compute **top-5 accuracy**: Is the correct person in top 5?
- Add **ROC curves** and **confusion matrices**

### 4. Real-Time Attendance System

- Connect to a webcam
- Detect faces in real-time
- Match against employee database
- Log attendance with timestamps

### 5. Advanced FAISS Indexes

For larger datasets (1000+ employees):

- Use `IndexIVFFlat` for faster approximate search
- Use `IndexHNSW` for even better speed/accuracy tradeoff
- Add GPU support with `faiss-gpu`

### 6. Augmentation & Robustness Testing

Test how well the system handles:

- Rotated images
- Partial occlusions (masks, glasses)
- Low-resolution images
- Different age ranges

---

## 🙏 Acknowledgements

This project was developed as part of the course:

> **Vector Database and Its Application in AI – Project 2.1**

### Libraries & Tools

- **[facenet-pytorch](https://github.com/timesler/facenet-pytorch)** — Pre-trained FaceNet (InceptionResnetV1) model trained on VGGFace2 dataset
- **[FAISS](https://github.com/facebookresearch/faiss)** — Fast vector similarity search library by Facebook AI Research
- **[PyTorch](https://pytorch.org/)** — Deep learning framework
- **[VGGFace2 Dataset](https://github.com/ox-vgg/vgg_face2)** — Large-scale face recognition dataset used for pre-training

### References

- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). _FaceNet: A unified embedding for face recognition and clustering._ CVPR 2015.
- Cao, Q., Shen, L., Xie, W., Parkhi, O. M., & Zisserman, A. (2018). _VGGFace2: A dataset for recognising faces across pose and age._ FG 2018.
- Johnson, J., Douze, M., & Jégou, H. (2019). _Billion-scale similarity search with GPUs._ IEEE Transactions on Big Data.

---

## 📝 Notes

- **Important:** Always use `python` (miniconda) instead of `python3` (system Python) to avoid module errors
- The first run of `facenet_app.py` will download the pre-trained model (~100MB)
- Both scripts save their indexes to disk, so subsequent runs will be faster
- For production use, consider adding face detection, alignment, and quality checks

---

## 📧 Contact

For questions or issues, please contact the project maintainer or refer to the course materials.

---

**Happy Face Recognition! 🎭**
