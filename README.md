# ECG CNN Classification Full-Stack App

A full-stack machine learning application for ECG image classification using a TensorFlow/Keras CNN model, a FastAPI backend, and a React + Vite frontend dashboard.

## Live Demo

Frontend deployment on Vercel:

https://vercel.com/amirthavanans-projects/ecg-classification/86RjsBgT2iWjF5cv6ASG7otjn1nj

## Project Objective

This project is designed to:
- accept ECG images uploaded by the user,
- preprocess the image to match the model input requirements,
- run the trained CNN model for inference,
- return the predicted class, confidence score, and probability distribution,
- display the result through a user-friendly web dashboard.

This project is intended for learning, demonstration, and decision-support purposes, not as a medical diagnosis system.

## Target Users

This project is intended for:
- students and learners exploring ML deployment,
- developers building full-stack AI applications,
- researchers and testing teams working with medical image classification prototypes,
- anyone learning how to connect a trained deep learning model to a web app.

## Tech Stack

- Frontend: React, Vite, Tailwind CSS
- Backend: FastAPI
- ML model: TensorFlow / Keras CNN
- Data handling: NumPy, Pillow
- API communication: Axios

## Project Structure

```text
ecg-cnn-fullstack/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── model_service.py
│   │   └── schemas.py
│   ├── models/
│   │   ├── classes.json
│   │   ├── ecg_cnn_model.keras
│   │   └── README_MODEL.txt
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── index.css
│       ├── main.jsx
│       └── components/
│           ├── ImageUploader.jsx
│           ├── PredictionCard.jsx
│           └── ProbabilityChart.jsx
├── README.md
└── .gitignore
```

## Model Details

The backend expects:
- an ECG image input,
- image resizing to 224 × 224,
- RGB conversion,
- normalization by dividing pixel values by 255,
- a final softmax output layer with class probabilities.

The model output classes must match the labels in `backend/models/classes.json`.

## Supported Output Classes

The current project is configured for class labels such as:
- AHB
- COVID-19
- HMI
- MI
- Normal

## Setup Instructions

### 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Place your trained model in:

```text
backend/models/ecg_cnn_model.keras
```

Ensure the label file exists:

```text
backend/models/classes.json
```

Run the backend:

```bash
uvicorn app.main:app --reload --port 8000
```

Backend URL:

```text
http://localhost:8000
```

Swagger docs:

```text
http://localhost:8000/docs
```

### 2. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open the Vite app in the browser, typically:

```text
http://localhost:5173
```

## API Endpoint

### Health check

```http
GET /api/health
```

### Prediction

```http
POST /api/predict
Content-Type: multipart/form-data
```

Body:
- `file`: ECG image file

Example response:

```json
{
  "predicted_class": "Normal",
  "confidence": 0.91,
  "probabilities": {
    "AHB": 0.01,
    "COVID-19": 0.02,
    "HMI": 0.03,
    "MI": 0.03,
    "Normal": 0.91
  },
  "model_input_size": [224, 224]
}
```

## Important Note

This project is a demonstration and research-oriented application. It should not be treated as a clinical diagnostic system or used as a substitute for medical judgment.

## License

This project is intended for educational and research use. Add an appropriate license if you plan to publish it publicly.

## Future Improvements

- add authentication,
- improve model accuracy,
- support more ECG conditions,
- add history tracking and user logs,
- deploy to cloud hosting,
- improve security and production readiness.
