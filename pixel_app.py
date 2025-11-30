"""
V3: Pixel-Based Face Recognition System
Uses raw pixel values (flattened) for similarity search with FAISS L2 distance.
"""

import os
import faiss
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================================
# DATA PREPARATION
# ========================================

dataset_path = 'Dataset'
image_paths = []
labels = []

for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.JPG', '.png', '.jpeg')):
        image_paths.append(os.path.join(dataset_path, filename))
        file_name = filename.split('.')[0]  # remove extensions
        label = file_name[7:]  # cut off "Avatar_" (first 7 chars)    
        labels.append(label)

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

print(f"✓ Loaded {len(df)} employee images")


# ========================================
# PIXEL-BASED VECTORIZATION (V3)
# ========================================

IMAGE_SIZE = 300
PIXEL_VECTOR_DIM = 300 * 300 * 3  # 270,000 dimensions

# FAISS (Facebook AI Similarity Search) is a fast vector-search library from Meta.
# We use it to store high-dimensional vectors and quickly find which stored vector
# is most similar to a new one. Instead of manually looping through every vector,
# FAISS builds optimized indexes so similarity search becomes extremely fast.

# Create FAISS index using L2 distance (Euclidean distance)
pixel_index = faiss.IndexFlatL2(PIXEL_VECTOR_DIM)
pixel_label_map = []


def image_to_vector(image_path):
    """
    Convert an image into a flattened pixel vector.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Flattened numpy array of shape (270000,) with normalized pixel values [0, 1]
    """
    img = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)
    
    # Convert the image into a NumPy array so we can work with the raw pixel values.
    # Image files (.jpg, .png) are just compressed files, but FAISS and ML models
    # require numerical data. A NumPy array turns the 300×300 RGB image into a
    # 300×300×3 matrix of pixel values that we can normalize, flatten, and use as a
    # vector for similarity search.
    
    # Handle grayscale images (convert to RGB)
    if len(img_array.shape) == 2:
        # For a 2D grayscale image with shape (H, W), stack 3 copies along the last axis
        # so it becomes (H, W, 3) = standard RGB format.
        img_array = np.stack((img_array,)*3, axis=-1)

    # Normalize pixel values to [0, 1]
    vector = img_array.astype('float32') / 255.0
    
    # Flatten from (300, 300, 3) to (270000,)
    return vector.flatten()


# Build the FAISS index
print("Building pixel-based FAISS index...")
for idx, row in df.iterrows():
    image_path = row['image_path']
    label = row['label']

    try:
        vector = image_to_vector(image_path)
        pixel_index.add(np.array([vector]))
        pixel_label_map.append(label)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Save FAISS index and labels
faiss.write_index(pixel_index, "employee_images.index")
np.save("label_map.npy", np.array(pixel_label_map))
print(f"✓ Saved pixel-based index with {len(pixel_label_map)} employees")


# ========================================
# PIXEL-BASED SEARCH ENGINE
# ========================================

def search_similar_images_pixel(query_image_path, k=5):
    """
    Search for similar employee images using pixel-based approach.
    
    Args:
        query_image_path: Path to the query image
        k: Number of top matches to return
        
    Returns:
        List of tuples (employee_name, distance)
    """
    index = faiss.read_index("employee_images.index")
    label_map = np.load("label_map.npy")
    query_vector = image_to_vector(query_image_path)
    distances, indices = index.search(np.array([query_vector]), k)
    
    results = []
    for i in range(len(indices[0])):
        employee_name = label_map[indices[0][i]]
        distance = distances[0][i]
        results.append((employee_name, distance))
    return results


def display_query_and_top_matches_pixel(query_image_path):
    """
    Display the query image and the top 5 matching images with distances.
    
    Args:
        query_image_path: Path to the query image
    """
    # Display the query image
    query_img = Image.open(query_image_path)
    query_img = query_img.resize((300, 300))

    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title("Query Image (Pixel-Based V3)")
    plt.axis("off")
    plt.show()

    # Search for similar images
    matches = search_similar_images_pixel(query_image_path, k=5)

    # Map results to image paths
    top_matches = []
    for name, distance in matches:
        # Find the corresponding image path in the dataframe
        img_path = df[df["label"] == name]["image_path"].values[0]
        top_matches.append((name, distance, img_path))

    # Plot top 5 images
    plt.figure(figsize=(15, 5))
    for i, (name, distance, img_path) in enumerate(top_matches):
        img = Image.open(img_path)
        img = img.resize((300, 300))

        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"{name}\nDist: {distance:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("V3: PIXEL-BASED FACE RECOGNITION")
    print("="*50 + "\n")
    
    # Test pixel-based search
    display_query_and_top_matches_pixel("Dataset/Avatar_Ngo_Ngoc.jpg")

