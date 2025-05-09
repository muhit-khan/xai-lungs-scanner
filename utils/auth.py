import os
import json
import logging
from config import Config

logger = logging.getLogger(__name__)

def validate_api_key(api_key):
    """
    Validate the API key against the stored valid keys.
    
    Args:
        api_key (str): The API key to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not api_key:
        return False
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(Config.API_KEYS_FILE), exist_ok=True)
        
        # Check if API keys file exists, create with default key if it doesn't
        if not os.path.exists(Config.API_KEYS_FILE):
            default_api_keys = {
                "valid_keys": ["test_key"]
            }
            with open(Config.API_KEYS_FILE, 'w') as f:
                json.dump(default_api_keys, f, indent=2)
            logger.info(f"Created default API keys file at {Config.API_KEYS_FILE}")
        
        # Load API keys
        with open(Config.API_KEYS_FILE, 'r') as f:
            api_keys_data = json.load(f)
            valid_keys = api_keys_data.get('valid_keys', [])
            
        return api_key in valid_keys
    
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        # Default to invalid on error
        return False
