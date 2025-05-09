import os
import json
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
from datetime import datetime
import uuid
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename

# Import modules
from models.model_loader import ModelLoader
from utils.preprocessing import preprocess_image
from utils.visualization import generate_gradcam, generate_placeholder_heatmap
from utils.auth import validate_api_key
from config import Config

# Configure logging
if not os.path.exists(Config.LOG_DIR):
    os.makedirs(Config.LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(Config.LOG_DIR, 'app.log'), 
            maxBytes=10485760, 
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize model loader
model_loader = None

# Ensure required directories exist
for directory in [Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER, Config.LOG_DIR, Config.MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# API key authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key', '') or request.args.get('api_key', '')
        
        if not api_key:
            return jsonify({'error': 'API key is required'}), 401
        
        if not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
            
        return f(*args, **kwargs)
    return decorated_function

@app.route('/', methods=['GET'])
def index():
    """Serve the API documentation."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    """
    Endpoint for making predictions on chest X-ray images.
    Requires an API key and an image file.
    """
    global model_loader
    
    # Check if file exists in the request
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        logger.warning("No file selected for uploading")
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        logger.warning(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg'}), 400
    
    try:
        # Save file with secure filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        basename, ext = os.path.splitext(filename)
        unique_filename = f"{basename}_{unique_id}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        logger.info(f"File saved: {filepath}")
        
        # Process image and make prediction
        if model_loader is None:
            logger.info("Loading models for prediction")
            model_loader = ModelLoader(generate_if_missing=True)
            
        # Load and preprocess image
        img = preprocess_image(filepath)
        
        # Run prediction
        try:
            results = model_loader.predict(img)
            using_random_predictions = results.get('is_random_prediction', False)
        except Exception as e:
            logger.warning(f"Prediction failed with error: {str(e)}. Using fallback prediction.")
            # Use fallback random prediction
            if model_loader is None:
                model_loader = ModelLoader(generate_if_missing=False)
            results = model_loader.generate_random_predictions(img)
            using_random_predictions = True
        
        # Generate explanations - ALWAYS do this for top predictions
        explanation_data = {}
        
        # Sort the predictions by score to get the top conditions
        prediction_scores = results['prediction_scores']
        sorted_indices = np.argsort(prediction_scores)[::-1]  # Descending order
        
        # Always generate explanations for the top 3 conditions
        for i in range(min(3, len(sorted_indices))):
            idx = sorted_indices[i]
            condition = results['labels'][idx]
            score = float(prediction_scores[idx])
            
            # Create output filename
            explanation_path = os.path.join(Config.RESULTS_FOLDER, f"{unique_id}_{condition}_explanation.jpg")
            
            # Always generate a placeholder heatmap
            success = generate_placeholder_heatmap(
                img,
                explanation_path,
                condition,
                score
            )
            
            # Only try GradCAM if we have models and we're not using random predictions
            if not using_random_predictions and hasattr(model_loader, 'base_models') and len(model_loader.base_models) > 0:
                # Try to replace placeholder with actual GradCAM
                grad_cam_path = os.path.join(Config.RESULTS_FOLDER, f"{unique_id}_{condition}_gradcam.jpg")
                grad_success = generate_gradcam(
                    model_loader.base_models,
                    img,
                    idx,
                    grad_cam_path,
                    model_loader.config
                )
                
                # If GradCAM was successful, use it instead of the placeholder
                if grad_success:
                    explanation_path = grad_cam_path
                    success = True
            
            if success:
                explanation_data[condition] = {
                    'score': score,
                    'explanation_path': os.path.relpath(explanation_path, start=Config.BASE_DIR),
                    'is_placeholder': using_random_predictions or not hasattr(model_loader, 'base_models') or len(model_loader.base_models) == 0
                }
                logger.info(f"Generated explanation for {condition} with score {score}")
            else:
                logger.warning(f"Failed to generate explanation for {condition}")
        
        # Log if we still have no explanations
        if not explanation_data:
            logger.error("No explanations were generated!")
            
            # Guaranteed fallback for explanation - use the top prediction no matter what
            top_idx = sorted_indices[0]
            top_condition = results['labels'][top_idx]
            top_score = float(prediction_scores[top_idx])
            
            # Forcefully create explanation for top condition
            fallback_path = os.path.join(Config.RESULTS_FOLDER, f"{unique_id}_fallback_explanation.jpg")
            
            try:
                # Simple direct image saving as last resort
                img_display = (img * 255).astype(np.uint8)
                # Add some text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    img_display,
                    f"{top_condition}: {top_score:.2f}",
                    (10, 30),
                    font, 0.7, (255, 0, 0), 2
                )
                cv2.putText(
                    img_display,
                    "Fallback Explanation",
                    (10, img_display.shape[0] - 10),
                    font, 0.6, (255, 0, 0), 1
                )
                
                # Save the image
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                cv2.imwrite(fallback_path, cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
                
                explanation_data[top_condition] = {
                    'score': top_score,
                    'explanation_path': os.path.relpath(fallback_path, start=Config.BASE_DIR),
                    'is_fallback': True
                }
                logger.info(f"Created fallback explanation for {top_condition}")
            except Exception as e:
                logger.error(f"Even fallback explanation generation failed: {str(e)}")
        
        # Prepare the response
        response = {
            'success': True,
            'predictions': [{
                'condition': results['labels'][i],
                'score': float(results['prediction_scores'][i]),
                'positive': bool(results['prediction_scores'][i] > Config.CLASSIFICATION_THRESHOLD)
            } for i in range(len(results['labels']))],
            'explanations': explanation_data,
            'timestamp': datetime.now().isoformat(),
            'note': "Using placeholder predictions (models not available)" if using_random_predictions else None
        }
        
        logger.info(f"Returning response with {len(explanation_data)} explanations")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/results/<filename>')
@require_api_key
def results(filename):
    """Serve generated result files like GradCAM visualizations."""
    return send_from_directory(Config.RESULTS_FOLDER, filename)

@app.route('/status', methods=['GET'])
@require_api_key
def status():
    """Return the status of the API and model availability."""
    global model_loader
    
    if model_loader is None:
        # Initialize model loader if not already done
        try:
            model_loader = ModelLoader(generate_if_missing=False)
        except Exception as e:
            logger.error(f"Error initializing model loader: {str(e)}")
            return jsonify({
                'status': 'degraded',
                'models_available': False,
                'error': str(e)
            }), 200
    
    # Check which models are available
    base_models_available = hasattr(model_loader, 'base_models') and len(model_loader.base_models) > 0
    decision_model_available = hasattr(model_loader, 'decision_model') and model_loader.decision_model is not None
    
    return jsonify({
        'status': 'operational' if base_models_available and decision_model_available else 'degraded',
        'base_models_available': base_models_available,
        'base_models_loaded': list(model_loader.base_models.keys()) if base_models_available else [],
        'decision_model_available': decision_model_available,
        'using_fallback': not (base_models_available and decision_model_available),
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {str(error)}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
