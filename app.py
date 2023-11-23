import os
from flask import Flask, render_template, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Define the upload and colorized folders
UPLOAD_FOLDER = '/Images/uploads'
COLORIZED_FOLDER = '/Images/colorized'

# To create the upload and colorized folders
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(COLORIZED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COLORIZED_FOLDER'] = COLORIZED_FOLDER

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def colorize_image(input_path, output_path):
    print("Loading model")
    DIR = os.path.dirname(os.path.abspath(__file__))
    PROTOTXT = os.path.join(DIR, "Model/colorization_deploy_v2.prototxt")
    POINTS = os.path.join(DIR, "Model/pts_in_hull.npy")
    MODEL = os.path.join(DIR, "Model/colorization_release_v2.caffemodel")

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    pts = np.load(POINTS)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    image = cv2.imread(input_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    cv2.imwrite(output_path, colorized)

    # Encode colorized image in Base64
    with open(output_path, "rb") as image_file:
        base64_encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return base64_encoded_image


# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    colorized_image = None

    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_path)
            uploaded_image = filename

            # Colorize the image
            output_path = os.path.join(app.config['COLORIZED_FOLDER'], filename)
            colorize_image(input_path, output_path)

            # Encode colorized image in Base64
            with open(output_path, "rb") as image_file:
                base64_encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            colorized_image = base64_encoded_image

    return render_template("index.html", uploaded_image=uploaded_image, colorized_image=colorized_image)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve colorized files
@app.route('/colorized/<filename>')
def colorized_file(filename):
    return send_from_directory(app.config['COLORIZED_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
