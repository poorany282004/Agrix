Agrix (AI-Powered Rice Crop Disease Detection System)


Team Information
Team Lead: Poorany Priyadharshiny T, Lakshmi Sangari S
Team Name: Agrix
College: Anna University

Agrix is an AI-powered Crop Protection System (CPS) designed to detect rice crop diseases using deep learning (CNN – MobileNetV2) and provide farmers with instant diagnosis, symptoms, and prevention steps.
The system includes:

Machine Learning model:-
1.Smart Image Scan (Camera + Upload)
2.Real-time prediction
3.Disease knowledge base
4.An interactive web dashboard
5.Assistive CPS pipeline for field deployment


Features:-

1.Upload or capture images directly from camera
2.CNN-based rice disease classification
3.Detection confidence + affected stage (Healthy / Partial / Full)
4.Auto-generated disease description, symptoms & prevention
5.Flask-based web dashboard


Project structure:-
Agrix/
│
├── app.py
├── model.py
├── fusion.py
├── train.py
├── advisory.json
├── requirements.txt
├── README.md
│
├── saved_model/
│   ├── agrix_model.keras
│   └── labels.txt
│
├── uploads/
│
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── diseases/
│
├── templates/
│   ├── cover.html
│   ├── scan.html
│   └── result.html
│
└── dataset/
    ├── train/
    └── val/



Model Architecture:
The Agrix AI uses MobileNetV2 Transfer Learning, fine-tuned on rice crop dataset.

Model pipeline:
Load MobileNetV2 (ImageNet)
Freeze base layers

Add:
GlobalAveragePooling
Dense(128, ReLU)
Dropout(0.2)
Dense(num_classes, Softmax)
Train on 224×224 crop leaf images



Installation & Setup:-
1️⃣ Clone the repository
git clone https://github.com/<your-username>/Agrix.git
cd Agrix

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Web App
python app.py

App runs on:

http://127.0.0.1:5000

Using Agrix:-
Option 1: Upload Image
Option 2: Capture using Camera



The model predicts:
1. Disease Name
2.Probability
3.Affected Stage (Healthy / Partial / Severe)
4.Symptoms
5.Prevention Steps
6.Auto-loaded disease info from /static/diseases/*.txt


Training Details Dataset:-
1.9 rice disease classes
2.Healthy class included
3.Train/Val split: 80/20
4.Preprocessing
5.Resize to 224×224
6.Normalize /255
7.Shuffle + Augmentation


Use Cases:-
1.Rice crop disease detection
2.Smart farming CPS
3.Mobile agriculture apps
4.Field assistant tools
5.Early warning systems

