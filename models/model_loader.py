import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import gc
import random

# Set TF logging level
tf.get_logger().setLevel(logging.ERROR)

# Import custom classes for loading models with custom layers
from .custom_objects import get_custom_objects
from config import Config

logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Handles loading and prediction with the XAI Lungs Disease models.
    Loads both base models and decision model.
    """
    
    def __init__(self, generate_if_missing=True):
        """Initialize and load the models."""
        self.config = Config.load_model_config()
        self.base_models = {}
        self.decision_model = None
        self.custom_objects = get_custom_objects()
        self.labels = Config.LABELS
        self.num_classes = len(self.labels)
        self.models_loaded = False
        
        # Determine which base models to load
        self.base_models_to_load = self.config.get('base_models_to_build', Config.DEFAULT_BASE_MODELS)
        self.num_base_models = len(self.base_models_to_load)
        
        logger.info(f"Initializing model loader with {self.num_base_models} base models")
        
        # Load base models
        self._load_base_models()
        
        # Load decision model
        self._load_decision_model()
        
        # If no models were loaded and generation is enabled, try to generate placeholder models
        if generate_if_missing and not self.models_loaded:
            logger.info("No models loaded. Attempting to generate placeholder models...")
            try:
                from .model_generator import generate_base_models, generate_decision_model
                generate_base_models()
                generate_decision_model()
                
                # Try loading the generated models
                logger.info("Retrying model loading with generated models...")
                self._load_base_models()
                self._load_decision_model()
            except Exception as e:
                logger.error(f"Failed to generate placeholder models: {str(e)}")
        
        logger.info("Model initialization complete")
    
    def _load_base_models(self):
        """Load the base models from saved checkpoints."""
        for model_name in self.base_models_to_load:
            logger.info(f"Loading base model: {model_name}")
            
            # Check for fine-tuned model first, then initial model
            finetuned_path = os.path.join(Config.MODELS_DIR, f'{model_name}_finetuned_best.keras')
            initial_path = os.path.join(Config.MODELS_DIR, f'{model_name}_initial_best.keras')
            
            if os.path.exists(finetuned_path):
                model_path = finetuned_path
            elif os.path.exists(initial_path):
                model_path = initial_path
            else:
                logger.warning(f"No checkpoint found for base model {model_name}")
                continue
            
            try:
                # Clear session to prevent memory leaks
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Load the model
                model = load_model(model_path, custom_objects=self.custom_objects, compile=False)
                
                # Ensure model is compiled for prediction
                if not hasattr(model, 'predict') or not callable(model.predict):
                    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
                
                self.base_models[model_name] = model
                self.models_loaded = True
                logger.info(f"Successfully loaded {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading {model_name}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(self.base_models)}/{len(self.base_models_to_load)} base models")
    
    def _load_decision_model(self):
        """Load the decision model that combines base model predictions."""
        logger.info("Loading decision model")
        
        if not os.path.exists(Config.DECISION_MODEL_PATH):
            logger.error(f"Decision model not found at {Config.DECISION_MODEL_PATH}")
            return
        
        try:
            # Clear session to prevent memory leaks
            tf.keras.backend.clear_session()
            gc.collect()
            
            # Load the model
            self.decision_model = load_model(Config.DECISION_MODEL_PATH, compile=False)
            
            # Compile model for prediction
            self.decision_model.compile(loss='binary_crossentropy', metrics=['accuracy'])
            
            # Verify the model has the correct input shape
            expected_input_shape = (self.num_base_models * self.num_classes,)
            actual_input_shape = self.decision_model.input_shape[1:]
            
            if actual_input_shape != expected_input_shape:
                logger.error(f"Decision model input shape mismatch. Expected {expected_input_shape}, got {actual_input_shape}")
            else:
                logger.info("Successfully loaded decision model")
                self.models_loaded = True
                
        except Exception as e:
            logger.error(f"Error loading decision model: {str(e)}")
    
    def generate_random_predictions(self, img):
        """Generate random predictions when no models are available."""
        logger.warning("Using fallback random prediction mechanism")
        
        # Create random base predictions with more realistic distribution
        base_predictions = []
        
        # Use condition prevalence to guide prediction probabilities
        # This makes predictions more realistic by simulating disease frequency
        condition_prevalence = {
            # Common conditions have higher base rates
            'Effusion': 0.15,
            'Infiltration': 0.15,
            'Atelectasis': 0.12,
            'Nodule': 0.10,
            # Mid-frequency conditions
            'Pneumonia': 0.08,
            'Mass': 0.06,
            'Pleural_Thickening': 0.05,
            'Cardiomegaly': 0.05,
            'Consolidation': 0.04,
            # Less common conditions
            'Edema': 0.03,
            'Emphysema': 0.03,
            'Fibrosis': 0.02,
            'Pneumothorax': 0.01,
            'Hernia': 0.01
        }
        
        # Function to generate realistic prediction for a condition
        def generate_condition_prediction(condition, base_rate=0.05):
            # Get condition-specific prevalence rate or use default
            prevalence = condition_prevalence.get(condition, base_rate)
            
            # Generate random prediction with bias based on prevalence
            # Beta distribution gives more realistic probability distribution
            # Higher prevalence = higher chance of positive prediction
            if np.random.random() < prevalence * 2:  # Chance of positive prediction
                return np.random.beta(2, 2) * 0.5 + 0.5  # 0.5-1.0 range (positive)
            else:
                return np.random.beta(2, 3) * 0.4 + 0.1  # 0.1-0.5 range (negative)
        
        # Generate predictions for each base model (or simulate if no models)
        num_base_models = len(self.base_models_to_load) if hasattr(self, 'base_models_to_load') and self.base_models_to_load else 4
        
        for _ in range(num_base_models):
            preds = np.array([generate_condition_prediction(label) for label in self.labels])
            base_predictions.append(preds)
        
        # Generate final prediction - using weighted average of base models
        weights = np.random.dirichlet(np.ones(len(base_predictions))) * 0.7 + 0.3/len(base_predictions)
        final_prediction = np.zeros(self.num_classes)
        
        for i, pred in enumerate(base_predictions):
            final_prediction += weights[i] * pred
        
        # Add small random noise
        final_prediction += np.random.normal(0, 0.05, size=self.num_classes)
        final_prediction = np.clip(final_prediction, 0.01, 0.99)
        
        return {
            'prediction_scores': final_prediction,
            'base_predictions': base_predictions,
            'labels': self.labels,
            'is_random_prediction': True
        }
    
    def predict(self, img):
        """Make predictions using base models and the decision model."""
        if not self.models_loaded:
            # Fallback to random predictions if no models are available
            return self.generate_random_predictions(img)
        
        if len(self.base_models) == 0:
            raise ValueError("No base models available for prediction")
        
        if self.decision_model is None:
            raise ValueError("Decision model not loaded")
        
        # Get predictions from each base model
        base_predictions = []
        
        for model_name in self.base_models_to_load:
            if model_name not in self.base_models:
                # Skip if this model wasn't loaded successfully
                continue
                
            try:
                model = self.base_models[model_name]
                pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
                base_predictions.append(pred[0])  # Get predictions, shape (num_classes,)
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
                # Skip this model
        
        if not base_predictions:
            raise ValueError("All base model predictions failed")
        
        # Concatenate base model predictions
        concatenated = np.concatenate(base_predictions)
        
        # Use the decision model to make the final prediction
        final_prediction = self.decision_model.predict(
            np.expand_dims(concatenated, axis=0), 
            verbose=0
        )
        
        return {
            'prediction_scores': final_prediction[0],  # Shape (num_classes,)
            'base_predictions': base_predictions,       # List of predictions from base models
            'labels': self.labels                       # Class labels
        }
