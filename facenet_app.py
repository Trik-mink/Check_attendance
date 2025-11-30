import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

print("Scanning dataset...")
DATASET_PATH = "Dataset"

image_paths = []
labels = []

for filename in os.listdir(DATASET_PATH):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        full_path = os.path.join(DATASET_PATH, filename)
        image_paths.append(full_path)

        file_name = filename.split(".")[0]   # "Avatar_Ngo_Ngoc"
        label = file_name[7:]               # remove "Avatar_"
        labels.append(label)

print(f"✓ Loaded {len(image_paths)} employee images")


# ================================
# 1. Load FaceNet model
# ================================
print("Loading FaceNet model...")
face_recognition_model = InceptionResnetV1(pretrained="vggface2").eval()
print("✓ FaceNet model loaded OK")

# image transform (same as in facet_mini.py)
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

def image_to_feature(image_path):
    """
    Convert an image file into a 512-dim FaceNet embedding (L2-normalized).
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        embedding = face_recognition_model(img_tensor)   # (1, 512)

    emb = embedding.squeeze().numpy().astype("float32") # (512,)
    # normalize so that inner product ≈ cosine similarity
    norm = np.linalg.norm(emb) + 1e-10
    emb /= norm
    return emb


# ================================
# 2. Build FAISS index
# ================================
print("Importing faiss and building index...")
import faiss

VECTOR_DIM = 512
index = faiss.IndexFlatIP(VECTOR_DIM)
label_map = []

for path, label in zip(image_paths, labels):
    try:
        feat = image_to_feature(path)
        # sanity check: all features must have correct dim
        if feat.shape[0] != VECTOR_DIM:
            print(f"⚠ Skipping {path}, wrong feature dim: {feat.shape}")
            continue

        index.add(np.array([feat]))
        label_map.append(label)
    except Exception as e:
        print(f"Error processing {path}: {e}")

print(f"✓ Added {len(label_map)} vectors to FAISS index")

# optional: save to disk
faiss.write_index(index, "facenet_features.index")
np.save("facenet_label_map.npy", np.array(label_map))


# ================================
# 3. Search API
# ================================
def search_similar_images_facenet(query_image_path, k=5):
    """
    Search for top-k similar employees using FaceNet embeddings and FAISS.
    """
    # We already have index + label_map in memory,
    # but for safety you could reload from disk if needed.
    # index = faiss.read_index("facenet_features.index")
    # label_map = np.load("facenet_label_map.npy")

    query_vec = image_to_feature(query_image_path)
    similarities, indices = index.search(np.array([query_vec]), k)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        name = label_map[idx]
        sim = similarities[0][i]
        results.append((name, sim))
    return results


# ================================
# 4. Display top-5 matches
# ================================
def display_query_and_top_matches_facenet(query_image_path, k=5):
    """
    Show the query image and the top-k similar images with similarity scores.
    """
    # show query
    query_img = Image.open(query_image_path).convert("RGB").resize((300, 300))
    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title("Query Image (FaceNet)")
    plt.axis("off")
    plt.show()

    # run search
    matches = search_similar_images_facenet(query_image_path, k=k)

    # map names back to paths
    top = []
    for name, sim in matches:
        # multiple images might share same label; just pick first match
        # based on the original image_paths/labels lists
        for path, lbl in zip(image_paths, labels):
            if lbl == name:
                top.append((name, sim, path))
                break

    plt.figure(figsize=(15, 5))
    for i, (name, sim, path) in enumerate(top):
        img = Image.open(path).convert("RGB").resize((300, 300))
        plt.subplot(1, k, i + 1)
        plt.imshow(img)
        plt.title(f"{name}\nSim: {sim:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ================================
# 5. Example usage
# ================================
if __name__ == "__main__":
    test_path = "Dataset/Avatar_Ngo_Ngoc.jpg"
    print(f"Running FaceNet search for: {test_path}")
    display_query_and_top_matches_facenet(test_path, k=5)
