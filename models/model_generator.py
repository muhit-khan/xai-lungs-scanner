import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
from config import Config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def load_training_history():
    """Load training history from the JSON file."""
    try:
        history_path = os.path.join(Config.MODELS_DIR, 'training_histories.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Training history file not found at {history_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading training history: {str(e)}")
        return {}

def load_decision_model_history():
    """Load decision model history."""
    try:
        history_path = os.path.join(Config.MODELS_DIR, 'decision_model_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Decision model history file not found at {history_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading decision model history: {str(e)}")
        return {}

def generate_base_models():
    """Generate simple base models for placeholders based on available data."""
    histories = load_training_history()
    
    for model_name in Config.DEFAULT_BASE_MODELS:
        if model_name in histories:
            logger.info(f"Generating placeholder model for {model_name}")
            
            # Define a simple model architecture
            inputs = layers.Input(shape=(*Config.IMG_SIZE, Config.IMG_CHANNELS))
            
            # Use different architectures based on model_name to simulate different models
            if model_name == "resnet50":
                x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
                
                # Simple residual block
                residual = x
                x = layers.Conv2D(64, (3, 3), padding="same")(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                x = layers.Conv2D(64, (3, 3), padding="same")(x)
                x = layers.BatchNormalization()(x)
                x = layers.add([x, residual])
                x = layers.Activation("relu")(x)
                
            elif model_name == "densenet121":
                x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding="same")(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
                
                # Simple dense block
                x1 = layers.Conv2D(32, (1, 1), padding="same")(x)
                x1 = layers.BatchNormalization()(x1)
                x1 = layers.Activation("relu")(x1)
                x1 = layers.Conv2D(32, (3, 3), padding="same")(x1)
                x = layers.Concatenate()([x, x1])
                
            elif model_name == "mobilenetv2":
                x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same")(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                
                # Simple inverted residual block
                expanded = layers.Conv2D(64, (1, 1), padding="same")(x)
                expanded = layers.BatchNormalization()(expanded)
                expanded = layers.Activation("relu")(expanded)
                expanded = layers.DepthwiseConv2D((3, 3), padding="same")(expanded)
                expanded = layers.BatchNormalization()(expanded)
                expanded = layers.Activation("relu")(expanded)
                x = layers.Conv2D(32, (1, 1), padding="same")(expanded)
                x = layers.BatchNormalization()(x)
                
            else:  # vgg16 or default
                x = layers.Conv2D(64, (3, 3), padding="same")(inputs)
                x = layers.Activation("relu")(x)
                x = layers.Conv2D(64, (3, 3), padding="same")(x)
                x = layers.Activation("relu")(x)
                x = layers.MaxPooling2D((2, 2))(x)
                x = layers.Conv2D(128, (3, 3), padding="same")(x)
                x = layers.Activation("relu")(x)
            
            # Common final layers
            x = layers.GlobalAveragePooling2D()(x)
            outputs = layers.Dense(len(Config.LABELS), activation="sigmoid")(x)
            
            model = models.Model(inputs, outputs)
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
            
            # Save the model
            model_path = os.path.join(Config.MODELS_DIR, f"{model_name}_finetuned_best.keras")
            model.save(model_path)
            logger.info(f"Saved placeholder model to {model_path}")
            
            # Clear session to prevent memory leaks
            tf.keras.backend.clear_session()
        else:
            logger.warning(f"No history data for {model_name}, skipping model generation")

def generate_decision_model():
    """Generate a simple decision model placeholder."""
    try:
        history = load_decision_model_history()
        if not history:
            logger.warning("No decision model history found, using default architecture")
        
        num_base_models = len(Config.DEFAULT_BASE_MODELS)
        input_shape = num_base_models * len(Config.LABELS)
        
        # Create a simple MLP model
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(len(Config.LABELS), activation="sigmoid")
        ])
        
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Save the model
        model_path = os.path.join(Config.MODELS_DIR, "decision_model_best.keras")
        model.save(model_path)
        logger.info(f"Saved placeholder decision model to {model_path}")
        
    except Exception as e:
        logger.error(f"Error generating decision model: {str(e)}")

if __name__ == "__main__":
    # Check if the models directory exists, create if it doesn't
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    
    logger.info("Starting model generation...")
    generate_base_models()
    generate_decision_model()
    logger.info("Model generation complete")
