from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load precomputed features and images
features = np.load('features.npy')
images = np.load('images.npy')  # images resized 224x224

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def prepare_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Prepare uploaded image
        user_img = prepare_image(file_path)
        user_feature = model.predict(user_img)

        # Compute similarity
        similarity = cosine_similarity(user_feature, features)[0]
        top_indices = similarity.argsort()[-6:][::-1]

        similar_images = []
        for idx in top_indices:
            # Save temporary image to static folder for display
            sim_img_path = os.path.join("static", f"sim_{idx}.jpg")
            cv2.imwrite(sim_img_path, images[idx])
            similar_images.append(sim_img_path)  # just store image path

        return render_template(
            "index.html",
            uploaded_image_url=file_path,
            similar_images=similar_images
        )

    return render_template("index.html", uploaded_image_url=None, similar_images=[])

if __name__ == "__main__":
    app.run(debug=True)
