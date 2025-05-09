# XAI Lungs Disease Prediction API

A Flask-based REST API for predicting lung diseases from chest X-ray images using Explainable AI (XAI) techniques.

## Overview

This API uses an ensemble of pretrained convolutional neural networks (CNNs) and a decision-making model to predict 14 different lung conditions from chest X-ray images. The API also provides visual explanations through Grad-CAM to highlight regions of interest in the images that contributed to predictions.

## Features

- **Multi-model Prediction**: Utilizes ResNet50, DenseNet121, MobileNetV2, and VGG16 as base models
- **Decision Making Model**: Combines predictions from base models for improved accuracy
- **Explainable AI**: Provides Grad-CAM visualizations to explain predictions
- **API Key Authentication**: Secure endpoints with API key validation
- **Comprehensive Documentation**: Interactive documentation at root endpoint

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/xai-lungs-disease.git
   cd xai-lungs-disease
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create required directories:

   ```bash
   mkdir -p uploads results logs models/pretrained
   ```

5. Place model files in the `models/pretrained` directory:
   - Base models: `[model_name]_finetuned_best.keras` or `[model_name]_initial_best.keras`
   - Decision model: `decision_model_best.keras`

## Usage

1. Start the API server:

   Using Python directly:

   ```bash
   python run_app.py
   ```

   Or, if you prefer to generate placeholder models first (useful for initial setup or if models are missing):

   ```bash
   python run_app.py --generate-models
   ```

   Alternatively, use Docker (see Docker Usage section below).

2. Access the API documentation:

   ```
   http://localhost:5000/
   ```

3. Send a prediction request:
   ```bash
   curl -X POST http://localhost:5000/predict \
     -H "X-API-Key: test_key" \
     -F "file=@/path/to/chest_xray.jpg"
   ```

## API Endpoints

- `GET /`: API documentation
- `POST /predict`: Submit a chest X-ray image for disease prediction
- `GET /results/{filename}`: Retrieve a result file (e.g., Grad-CAM visualization)
- `GET /status`: Check API and model status

## Docker Usage

This application can be run inside a Docker container.

### Prerequisites

- Docker installed on your system.

### Building the Docker Image

1.  Navigate to the project root directory (where the `Dockerfile` is located).
2.  Build the Docker image:

    ```bash
    docker build -t xai-lungs-scanner .
    ```

### Running the Docker Container

1.  Run the container, mapping the port and mounting volumes for persistent data (uploads, results, logs, models):

    ```bash
    docker run -d \
      -p 5000:5000 \
      -v $(pwd)/uploads:/app/uploads \
      -v $(pwd)/results:/app/results \
      -v $(pwd)/logs:/app/logs \
      -v $(pwd)/models/pretrained:/app/models/pretrained \
      -v $(pwd)/config:/app/config \
      --name xai-scanner-app \
      xai-lungs-scanner
    ```

    **Explanation of options:**

    - `-d`: Run in detached mode (background).
    - `-p 5000:5000`: Map port 5000 on the host to port 5000 in the container.
    - `-v $(pwd)/uploads:/app/uploads`: Mount the local `uploads` directory to `/app/uploads` in the container.
    - `-v $(pwd)/results:/app/results`: Mount the local `results` directory to `/app/results` in the container.
    - `-v $(pwd)/logs:/app/logs`: Mount the local `logs` directory to `/app/logs` in the container.
    - `-v $(pwd)/models/pretrained:/app/models/pretrained`: Mount your local pretrained models into the container.
    - `-v $(pwd)/config:/app/config`: Mount your local config directory (for `api_keys.json`) into the container.
    - `--name xai-scanner-app`: Assign a name to the container for easier management.
    - `xai-lungs-scanner`: The name of the image to use.

    **Note for Windows users:** Replace `$(pwd)` with `%cd%` or the absolute path to your project directory.

2.  The API will be accessible at `http://localhost:5000`.

### Managing the Container

- **View logs:**
  ```bash
  docker logs xai-scanner-app
  ```
- **Stop the container:**
  ```bash
  docker stop xai-scanner-app
  ```
- **Start a stopped container:**
  ```bash
  docker start xai-scanner-app
  ```
- **Remove the container (after stopping):**
  ```bash
  docker rm xai-scanner-app
  ```

### Accessing Shell inside the Container (for debugging)

    ```bash
    docker exec -it xai-scanner-app /bin/bash
    ```

## Project Structure

```
.
├── app.py                  # Main Flask application file
├── config.py               # Configuration settings
├── config/
│   └── api_keys.json       # API keys for authentication
├── models/
│   ├── custom_objects.py   # Custom Keras layers
│   ├── model_config.json   # Model configuration
│   ├── model_loader.py     # Model loading and prediction
│   └── pretrained/         # Pretrained model files
├── templates/
│   └── index.html          # API documentation
├── uploads/                # Uploaded image files
├── results/                # Generated result files (e.g., Grad-CAM)
├── utils/
│   ├── auth.py             # Authentication utilities
│   ├── preprocessing.py    # Image preprocessing
│   └── visualization.py    # Grad-CAM visualization
└── logs/                   # Application logs
```

## Requirements

- Python 3.8+
- TensorFlow 2.8+
- Flask 2.0+
- NumPy
- OpenCV
- Matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NIH Chest X-ray dataset for research and development
- TensorFlow and Keras teams for the deep learning framework
- Grad-CAM authors for the visualization technique
