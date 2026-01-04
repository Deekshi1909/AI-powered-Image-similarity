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


The workflow of the Image Similarity System begins with collecting images and placing them inside the dataset folder, which serves as the image database. These images are then processed by the feature extraction script, which reads each image, resizes it to a standard input size, and passes it through a pre-trained ResNet50 model to extract deep visual features. The extracted feature vectors are stored in a file for fast similarity computation, while the resized images are saved separately to enable quick display of results. Once feature extraction is completed, the main application file loads the saved feature files and initializes the deep learning model. A web interface is rendered using an HTML template, allowing users to upload an image through the browser. The uploaded image is temporarily saved in a static folder and processed using the same feature extraction pipeline to maintain consistency. The system then compares the uploaded image features with the dataset features using cosine similarity, identifies the most similar images, and finally displays the top matching images on the frontend. This flow ensures a clear separation between data preparation, feature extraction, backend processing, and frontend presentation, which is reflected in the project’s file and folder organization.


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
You can use **any general image dataset** that contains diverse categories (objects, people, nature, places, etc.).
- Recommended datasets:
  - **Caltech-101**
  - **Open Images Dataset (subset)**
  - **Custom image collection**

---

## Requirements

- Python 3.10+  
- Packages:
```bash
pip install flask numpy opencv-python tensorflow scikit-learn


