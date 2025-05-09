import cv2
import numpy as np
import tensorflow as tf
import logging
from config import Config

logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        np.ndarray: Preprocessed image with shape (224, 224, 3) and values in [0,1]
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, Config.IMG_SIZE)
        
        # Convert to float and normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        
        # Ensure shape is correct (224, 224, 3)
        assert img.shape == (*Config.IMG_SIZE, Config.IMG_CHANNELS)
        
        return img
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        # Return a blank image as fallback
        return np.zeros((*Config.IMG_SIZE, Config.IMG_CHANNELS), dtype=np.float32)
