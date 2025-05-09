import os
import sys
import logging
import json
from flask import Flask, request, jsonify
from werkzeug.datastructures import FileStorage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_prediction_endpoint():
    """Test the prediction endpoint with a sample image."""
    try:
        # Import the app
        from app import app, model_loader
        
        # Create a test client
        client = app.test_client()
        
        # Find a test image
        test_images_dir = 'test_images'
        if not os.path.exists(test_images_dir):
            os.makedirs(test_images_dir)
            logger.info(f"Created test images directory: {test_images_dir}")
            logger.warning("Please add test images to the test_images directory")
            return
        
        # Get the first image file
        image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            logger.warning(f"No image files found in {test_images_dir}")
            return
        
        test_image = os.path.join(test_images_dir, image_files[0])
        logger.info(f"Using test image: {test_image}")
        
        # Create FileStorage object
        with open(test_image, 'rb') as f:
            file_content = f.read()
        
        file = FileStorage(
            stream=open(test_image, 'rb'),
            filename=os.path.basename(test_image),
            content_type='image/jpeg' if test_image.lower().endswith('.jpg') or test_image.lower().endswith('.jpeg') else 'image/png'
        )
        
        # Make a request to the /predict endpoint
        response = client.post(
            '/predict',
            data={'file': file},
            headers={'X-API-Key': 'test_key'}
        )
        
        # Check the response
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            data = json.loads(response.data)
            logger.info(f"Predictions: {len(data['predictions'])} conditions")
            logger.info(f"Explanations: {len(data['explanations'])} conditions")
            
            # Print top 3 predictions
            sorted_predictions = sorted(data['predictions'], key=lambda x: x['score'], reverse=True)
            logger.info("Top 3 predictions:")
            for i, pred in enumerate(sorted_predictions[:3]):
                logger.info(f"  {i+1}. {pred['condition']}: {pred['score']:.4f} ({'Positive' if pred['positive'] else 'Negative'})")
            
            # Check if there are explanations
            if not data['explanations']:
                logger.error("No explanations were generated!")
            else:
                logger.info("Explanations were generated for:")
                for condition, explanation in data['explanations'].items():
                    logger.info(f"  - {condition}: {explanation}")
        else:
            logger.error(f"Request failed: {response.data}")
    
    except Exception as e:
        logger.error(f"Error testing prediction endpoint: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Run the test
    test_prediction_endpoint()
