# 🫁 XAI Lungs Scanner

**Predict 14 lung diseases from chest X-ray images using Deep Learning and Explainable AI (XAI) with a secure Flask-based REST API.**

---

## 🧠 Overview

The **XAI Lungs Scanner** is a medical imaging tool tailored for researchers and developers. It uses **deep learning ensemble models** to detect **14 common lung conditions** and provides **Grad-CAM visualizations** to explain predictions.

Designed for transparency and trust, it enables fast, interpretable diagnostics while being easy to deploy and integrate.

---

## 🚀 Features at a Glance

| Feature                      | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| 🧬 **Multi-Disease Support** | Detects 14 lung diseases from chest X-rays.                                 |
| 🧠 **Explainable AI**        | Generates Grad-CAM heatmaps to visualize model decisions.                   |
| 🧩 **Model Ensemble**        | Combines multiple CNNs (ResNet50, DenseNet121, etc.) with a decision model. |
| 🔐 **Secure API Access**     | Secured using API key authentication.                                       |
| 📦 **Dockerized**            | Fully containerized for deployment ease.                                    |
| 🔄 **Fallback Mode**         | Works with placeholder outputs if real models are unavailable.              |
| 📈 **Status Monitoring**     | `/status` endpoint to check model availability and system health.           |

---

## 🏗️ Tech Stack

- **Backend:** Flask (Python)
- **Model Serving:** TensorFlow + Keras
- **XAI Method:** Grad-CAM
- **Containerization:** Docker
- **Server:** Gunicorn (via `run_app.py`)
- **Dependencies:** OpenCV, Pillow, NumPy
- **Python Version:** 3.9+

---

## 🛠️ Setup Guide

### 🔧 Prerequisites

- Python 3.9+
- Git
- Docker (optional, for deployment)

### 📥 Installation Steps

```bash
# Clone the repository
git clone <https://github.com/muhit-khan/xai-lungs-scanner.git>
cd xai-lungs-scanner

# Create virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Create necessary folders
mkdir -p uploads results logs models/pretrained config
```

### 📦 Place Pretrained Models

Place your `.keras` or `.h5` files in `models/pretrained/`:

- CNN base models (e.g., `resnet50_finetuned_best.keras`)
- Decision model (e.g., `decision_model_best.keras`)

Update `models/model_config.json` if needed.

### 🔐 API Key Setup

Create or edit `config/api_keys.json`:

```json
{
  "valid_keys": ["your_secret_api_key_1", "test_key"]
}
```

> If this file is missing, it will be auto-generated with a default `"test_key"` on first API call.

---

## ▶️ Running the Application

### 🖥️ Run Locally

```bash
python run_app.py
```

Optional arguments:

```bash
# Generate placeholder models and run
python run_app.py --generate-models

# Custom host and port
python run_app.py --host 0.0.0.0 --port 5001
```

Visit: [http://localhost:5000](http://localhost:5000)

---

### 🐳 Run with Docker

#### Build Docker Image

```bash
docker build -t xai-lungs-scanner .
```

or,

#### Pull Docker Image From Docker Hub

```bash
docker pull muhitkhan/xai_lungs_scanner:latest
```

#### Run Docker Container

```bash
docker run -p 5000:5000 muhitkhan/xai_lungs_scanner:latest
```

> On macOS/Linux, replace `%cd%` with `$(pwd)`

#### Docker Commands

- View logs: `docker logs xai-scanner-app`
- Stop container: `docker stop xai-scanner-app`
- Start container: `docker start xai-scanner-app`
- Remove container: `docker rm xai-scanner-app`

---

## 📡 API Reference

All endpoints require authentication via:

- Header: `X-API-Key: your_api_key_here`
  _OR_
- Query: `?api_key=your_api_key_here`

---

### `GET /`

- **Description:** Returns interactive API documentation in HTML format.
- **Use:** Open in browser: [http://localhost:5000](http://localhost:5000)

---

### `POST /predict`

- **Purpose:** Predict lung diseases and generate Grad-CAMs.
- **Content Type:** `multipart/form-data`
- **File Field:** `file` (image: PNG/JPEG/JPG)

#### Example (cURL)

```bash
curl -X POST http://localhost:5000/predict \
  -H "X-API-Key: test_key" \
  -F "file=@/path/to/chest_xray.jpg"
```

#### Example Response

```json
{
  "explanations": {
    "Mass": {
      "explanation_path": "results/21bc02ee-6874-4a81-9ba0-ed3517761a43_Mass_explanation.jpg",
      "is_placeholder": false,
      "score": 0.5705410242080688
    },
    "Nodule": {
      "explanation_path": "results/21bc02ee-6874-4a81-9ba0-ed3517761a43_Nodule_explanation.jpg",
      "is_placeholder": false,
      "score": 0.5869627594947815
    },
    "Pneumonia": {
      "explanation_path": "results/21bc02ee-6874-4a81-9ba0-ed3517761a43_Pneumonia_explanation.jpg",
      "is_placeholder": false,
      "score": 0.573369026184082
    }
  },
  "note": null,
  "predictions": [
    {
      "condition": "Atelectasis",
      "positive": true,
      "score": 0.5374575257301331
    },
    {
      "condition": "Cardiomegaly",
      "positive": true,
      "score": 0.5286591053009033
    },
    {
      "condition": "Consolidation",
      "positive": false,
      "score": 0.4373806416988373
    },
    {
      "condition": "Edema",
      "positive": false,
      "score": 0.492987722158432
    },
    {
      "condition": "Effusion",
      "positive": false,
      "score": 0.48966044187545776
    },
    {
      "condition": "Emphysema",
      "positive": false,
      "score": 0.33150017261505127
    },
    {
      "condition": "Fibrosis",
      "positive": false,
      "score": 0.42724230885505676
    },
    {
      "condition": "Hernia",
      "positive": false,
      "score": 0.45632579922676086
    },
    {
      "condition": "Infiltration",
      "positive": true,
      "score": 0.5135630369186401
    },
    {
      "condition": "Mass",
      "positive": true,
      "score": 0.5705410242080688
    },
    {
      "condition": "Nodule",
      "positive": true,
      "score": 0.5869627594947815
    },
    {
      "condition": "Pleural_Thickening",
      "positive": true,
      "score": 0.5169267058372498
    },
    {
      "condition": "Pneumonia",
      "positive": true,
      "score": 0.573369026184082
    },
    {
      "condition": "Pneumothorax",
      "positive": true,
      "score": 0.553454577922821
    }
  ],
  "success": true,
  "timestamp": "2025-05-09T08:09:20.671405"
}
```

---

### `GET /results/{filename}`

- **Use:** Fetch Grad-CAM images.
- **Example:**

```bash
curl -X GET "http://localhost:5000/results/uuid_Pneumonia_gradcam.jpg?api_key=test_key" \
  --output pneumonia_cam.jpg
```

---

### `GET /status`

- **Use:** Check system health and model availability.

#### Example (cURL)

```bash
curl -X GET "http://localhost:5000/status?api_key=test_key"
```

#### Sample Output

```json
{
  "status": "operational",
  "base_models_available": true,
  "base_models_loaded": ["resnet50", "densenet121", "mobilenetv2", "vgg16"],
  "decision_model_available": true,
  "using_fallback": false
}
```

---

## 🧾 List of Detectable Lung Conditions

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Effusion
6. Emphysema
7. Fibrosis
8. Hernia
9. Infiltration
10. Mass
11. Nodule
12. Pleural Thickening
13. Pneumonia
14. Pneumothorax

---

## 📁 Project Directory Structure

```
xai-lungs-scanner/
├── app.py
├── run_app.py
├── Dockerfile
├── requirements.txt
├── config/
│   └── api_keys.json
├── models/
│   ├── model_loader.py
│   ├── model_config.json
│   ├── custom_objects.py
│   └── pretrained/
├── templates/
│   └── index.html
├── utils/
│   ├── auth.py
│   ├── preprocessing.py
│   └── visualization.py
├── uploads/
├── results/
├── logs/
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 📄 License

Licensed under the MIT License.
See the full license in the [LICENSE](LICENSE) file.

---

## 🤝 Contributing

We welcome all contributors!

- Open an issue with suggestions or bugs.
- Fork the repo and create a pull request.
- Follow the project’s coding style and include tests if possible.

---

## 🙏 Acknowledgments

- Powered by **TensorFlow** and **Keras**
- Grad-CAM for transparent model explanations
- Inspired by the need for explainable AI in healthcare
- Dataset inspiration: NIH Chest X-ray Dataset (optional)
