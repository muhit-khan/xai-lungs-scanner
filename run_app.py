import os
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run XAI Lungs Disease API with options')
    parser.add_argument('--generate-models', action='store_true', help='Generate placeholder models before starting')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the API on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the API to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    if args.generate_models:
        logger.info("Generating placeholder models...")
        try:
            from models.model_generator import generate_base_models, generate_decision_model
            generate_base_models()
            generate_decision_model()
            logger.info("Model generation complete")
        except Exception as e:
            logger.error(f"Error generating models: {str(e)}")
    
    # Set environment variables for Flask
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['PORT'] = str(args.port)
    os.environ['HOST'] = args.host
    if args.debug:
        os.environ['DEBUG'] = 'True'
    
    # Start the Flask application
    logger.info(f"Starting Flask app on {args.host}:{args.port}")
    from app import app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
