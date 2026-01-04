# AI-Powered Image Similarity System

This project is an **AI-based Image Similarity System** built with **Flask** and **Deep Learning**.  

It allows users to **upload any image** and instantly see the **top visually similar images** from a precomputed dataset. The system uses **ResNet50** to extract image features and **cosine similarity** to compare images.

---

## Key Features

- **Visual similarity search**: Find similar images to any uploaded image.  
- **Dark-themed modern UI**: Clean and responsive design.  
- **Top 6 similar images**: Displayed in **2 columns × 3 rows** for easy viewing.  
- **Precomputed features**: Optimized for faster search.  
- **Purely visual**: No labels required — works with any image.  

---
## 🗂️ Project Structure

AI-powered-Image-similarity/
│
├── app.py
├── extract_features.py
├── templates/
│ └── index.html
│
├── static/
│ └── uploads/
│
├── features.npy
├── images.npy
│
├── .gitignore
├── README.md

---

## How It Works

1. **Dataset preparation**: Images are resized to 224x224 and features are extracted using ResNet50.  
2. **Upload an image**: Users select an image file via the web interface.  
3. **Feature extraction**: The uploaded image is processed using the same ResNet50 model.  
4. **Similarity search**: Cosine similarity is computed between the uploaded image and the dataset features.  
5. **Display results**: The top 6 similar images are displayed in a responsive 2×3 grid.  

---

## Dataset

This project uses a large image dataset for feature extraction.
Due to size constraints, the dataset is not included in this repository.
You can use Open Images, ImageNet, or any custom image dataset.

---

## Requirements

- Python 3.10+  
- Packages:
```bash
pip install flask numpy opencv-python tensorflow scikit-learn

