import os
import json

class Config:
    # Base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Server settings
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Model settings
    MODELS_DIR = os.path.join(BASE_DIR, 'models', 'pretrained')
    MODEL_CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'model_config.json')
    DECISION_MODEL_PATH = os.path.join(MODELS_DIR, 'decision_model_best.keras')
    
    # Authentication
    API_KEYS_FILE = os.path.join(BASE_DIR, 'config', 'api_keys.json')
    
    # Model parameters
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3
    CLASSIFICATION_THRESHOLD = 0.5
    
    # Always generate explanations for top N predictions regardless of score
    ALWAYS_EXPLAIN_TOP_N = 3
    
    # Generate explanations for any prediction with score above this threshold
    EXPLANATION_THRESHOLD = 0.3
    
    # Labels for prediction classes
    LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
        'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
    ]
    
    # Base model settings (these will be loaded from model_config.json)
    DEFAULT_BASE_MODELS = ['resnet50', 'densenet121', 'mobilenetv2', 'vgg16']
    
    # GradCAM settings
    GRAD_CAM_LAYER_MAP = {
        'resnet50': None,    # Auto-detect
        'densenet121': None, # Auto-detect
        'mobilenetv2': None, # Auto-detect
        'vgg16': None        # Auto-detect
    }
    HEATMAP_ALPHA = 0.5
    HEATMAP_COLORMAP = 'jet'
    
    # Load model configuration if exists
    @classmethod
    def load_model_config(cls):
        if os.path.exists(cls.MODEL_CONFIG_PATH):
            try:
                with open(cls.MODEL_CONFIG_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading model configuration: {e}")
        return {}
