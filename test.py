import cv2
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import torchxrayvision as xrv
import os
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import requests
import pydicom

ORTHANC_URL = "http://127.0.0.1:8042"  # Replace with your Orthanc server URL

def send_dicom_to_orthanc(dicom_path):
    # Read DICOM file
    dicom_data = pydicom.dcmread(dicom_path, force=True)

    # Send DICOM instance to Orthanc
    response = requests.post(f"{ORTHANC_URL}/instances", files={'file': dicom_data.tobytes()})

    if response.status_code == 200:
        print("DICOM instance sent to Orthanc successfully!")
        return response.json()["ID"]
    else:
        print(f"Failed to send DICOM instance to Orthanc. Status code: {response.status_code}")
        return None

app = Flask(__name__, static_folder='static')
CORS(app)
folder_path = "static/output_files"

UPLOAD_FOLDER = 'static/source_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_file_paths(file_names):
    return [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in file_names]

def load_model():
    model_name = "densenet121-res224-nih"
    model = xrv.models.DenseNet(weights=model_name)
    return model

def model_prediction(model, img):
    img = cv2.resize(img, (224, 224))
    img = xrv.datasets.normalize(img, 255)

    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    img = img[None, :, :]

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    img = transform(img)

    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        preds = model(img).cpu()
        output = {
            k: float(v)
            for k, v in zip(xrv.datasets.default_pathologies, preds[0].detach().numpy())
        }

    filtered_output = {k: v for k, v in output.items() if k in ['Infiltration', 'Atelectasis', 'Consolidation', 'Effusion', 'Nodule', 'Cardiomegaly', 'Mass']}

    sorted_output = dict(sorted(filtered_output.items(), key=lambda item: item[1], reverse=True))
    top2_output = dict(list(sorted_output.items())[:2])
    print(top2_output)
    return top2_output

def image_label(img, labels_dict, output_path):
    img_height, img_width, _ = img.shape
    x_pos = 0.2 * img_width
    y_pos = 0.1 * img_height
    for label, label_prob in labels_dict.items():
        print(label, label_prob)
        color = (0, 0, 255)
        thickness = 6
        cv2.putText(img, f'{label} : {label_prob:.2f}', (int(x_pos), int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (52, 152, 219), 4)
        y_pos = y_pos + 50
    cv2.imwrite(output_path, img)

def single_inference_image(img_path):
    img = cv2.imread(img_path)
    filename = os.path.basename(img_path)

    output_folder = 'static/output_files'
    output_path = f'{output_folder}/{filename}'

    results = model_prediction(model, img)
    image_label(img, results, output_path)

    print(output_path)
    return output_path

model = load_model()

@app.route('/', methods=['GET', 'POST'])
def status():
    return "Welcome to AIR"

@app.route('/process_files', methods=['POST'])
def main():
    files = request.files.getlist('files')
    print(files)
    file_paths = []
    output_paths = []
    for file in files:
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)

    for filepath in file_paths:
        output_path = single_inference_image(filepath)
        output_paths.append(output_path)

        # Send DICOM to Orthanc
        instance_id = send_dicom_to_orthanc(filepath)
        print(f"Instance ID in Orthanc: {instance_id}")

    return jsonify({'files': output_paths})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=5001, debug=True)
