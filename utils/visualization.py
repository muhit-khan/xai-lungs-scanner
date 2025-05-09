import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)

def find_last_conv_layer(model):
    """
    Find the name of the last conv or activation layer before GAP/Flatten.
    
    Args:
        model: The model to analyze
        
    Returns:
        str: Name of the last convolutional layer
    """
    target_layer = None
    from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Conv2D, Activation
    
    # Iterate backwards from the second-to-last layer
    for layer in reversed(model.layers[:-1]):
        # Stop if we hit Global Pooling, Flatten, or Dense (the head starts here)
        if isinstance(layer, (GlobalAveragePooling2D, Flatten, Dense)):
             break # We've gone past the convolutional base

        # Check if the layer's output is 4D (batch, height, width, channels)
        if hasattr(layer, 'output_shape') and isinstance(layer.output_shape, tuple) and len(layer.output_shape) == 4:
             # Prioritize layers typically found at the end of conv blocks
             is_conv_type = isinstance(layer, (Conv2D, Activation))
             has_conv_name = any(kw in layer.name.lower() for kw in [
                 'conv', 'relu', 'add', 'bn', 'block', 'project', 'expand', 
                 'out', 'attention', 'refined'
             ])
             
             if is_conv_type or has_conv_name:
                   target_layer = layer

    if target_layer:
        return target_layer.name

    # Fallback: Try the last layer with a 4D output
    for layer in reversed(model.layers[:-1]):
        if hasattr(layer, 'output_shape') and isinstance(layer.output_shape, tuple) and len(layer.output_shape) == 4:
            return layer.name
    
    return None

def generate_gradcam(base_models, img_array, class_index, output_path, config):
    """
    Generate Grad-CAM visualization for an image using base models.
    
    Args:
        base_models (dict): Dictionary of loaded base models
        img_array (np.ndarray): Preprocessed image
        class_index (int): Index of the target class
        output_path (str): Path to save the visualization
        config (Config): Configuration object
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # If no base models are available, return failure
        if not base_models or len(base_models) == 0:
            logger.error("No base models available for GradCAM")
            return False
            
        # Select a model that's likely to provide good visualizations
        # Prefer models with known good performance for GradCAM like ResNet or VGG
        model_priority = ['resnet50', 'vgg16', 'densenet121', 'mobilenetv2']
        selected_model = None
        model_name = None
        
        for name in model_priority:
            if name in base_models:
                selected_model = base_models[name]
                model_name = name
                break
                
        if selected_model is None:
            # Use the first available model if none of the preferred ones are available
            if base_models:
                model_name = next(iter(base_models))
                selected_model = base_models[model_name]
            else:
                logger.error("No base models available for GradCAM")
                return False
        
        logger.info(f"Generating GradCAM using {model_name}")
        
        # Get target layer
        layer_name = config.get('GRAD_CAM_LAYER_MAP', {}).get(model_name)
        if not layer_name:
            layer_name = find_last_conv_layer(selected_model)
            if not layer_name:
                logger.error(f"Could not find suitable layer for GradCAM in {model_name}")
                return False
        
        logger.info(f"Using layer {layer_name} for GradCAM")
        
        # Create GradCAM model
        grad_model = Model(
            inputs=[selected_model.inputs],
            outputs=[
                selected_model.get_layer(layer_name).output, 
                selected_model.output
            ]
        )
        
        # Prepare input
        img_tensor = tf.cast(np.expand_dims(img_array, axis=0), tf.float32)
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            class_output = predictions[:, class_index]
            
        # Get gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Pool gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps with gradients
        conv_outputs_squeezed = conv_outputs[0]
        heatmap = tf.reduce_sum(
            tf.multiply(conv_outputs_squeezed, pooled_grads), 
            axis=-1
        )
        
        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        
        # Apply colormap
        cmap = cm.get_cmap(config.HEATMAP_COLORMAP)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Take RGB channels only
        
        # Convert image and heatmap for blending
        img_display = (img_array * 255).astype(np.uint8)
        heatmap_display = (heatmap_colored * 255).astype(np.uint8)
        
        # Overlay heatmap on image
        superimposed = cv2.addWeighted(
            img_display, 
            1.0 - config.HEATMAP_ALPHA, 
            heatmap_display, 
            config.HEATMAP_ALPHA, 
            0
        )
        
        # Save the visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating GradCAM visualization: {str(e)}", exc_info=True)
        return False

def generate_placeholder_heatmap(img_array, output_path, condition_name, score):
    """
    Generate a placeholder heatmap visualization when models are not available.
    
    Args:
        img_array (np.ndarray): Preprocessed image
        output_path (str): Path to save the visualization
        condition_name (str): Name of the condition being visualized
        score (float): Prediction score for the condition
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert image to display format
        img_display = (img_array * 255).astype(np.uint8)
        
        # Create a heatmap based on the condition and score
        # Higher score = more intense heatmap
        heatmap = np.zeros((img_array.shape[0], img_array.shape[1]))
        
        # Generate semi-realistic heatmap patterns
        # Different patterns for different conditions
        if "pneumonia" in condition_name.lower() or "consolidation" in condition_name.lower():
            # Often affects lower lungs
            center_x = img_array.shape[0] // 2
            center_y = int(img_array.shape[1] * 0.6)  # Lower part
            radius = int(min(img_array.shape[0], img_array.shape[1]) * 0.3)
            
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < radius:
                        heatmap[i, j] = max(heatmap[i, j], (1 - dist/radius) * score)
        
        elif "effusion" in condition_name.lower() or "edema" in condition_name.lower():
            # Often seen at the base of lungs
            for i in range(img_array.shape[0]):
                # Gradient from bottom to middle
                h_val = max(0, 1 - (i / (img_array.shape[0] * 0.7)))
                for j in range(img_array.shape[1]):
                    heatmap[i, j] = h_val * score
        
        elif "cardiomegaly" in condition_name.lower():
            # Center of chest
            center_x = img_array.shape[0] // 2
            center_y = img_array.shape[1] // 2
            radius = int(min(img_array.shape[0], img_array.shape[1]) * 0.25)
            
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < radius:
                        heatmap[i, j] = max(heatmap[i, j], (1 - dist/radius) * score)
        
        elif "pneumothorax" in condition_name.lower():
            # Often one side of the lung
            side = 1 if np.random.random() > 0.5 else -1
            center_x = img_array.shape[0] // 2
            center_y = img_array.shape[1] // 2 + (side * img_array.shape[1] // 4)
            
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < img_array.shape[0] // 3:
                        heatmap[i, j] = max(heatmap[i, j], (1 - dist/(img_array.shape[0]//3)) * score)
        
        else:
            # Generic pattern: multiple focused areas
            num_spots = 3
            for _ in range(num_spots):
                x = np.random.randint(img_array.shape[0]//4, img_array.shape[0]*3//4)
                y = np.random.randint(img_array.shape[1]//4, img_array.shape[1]*3//4)
                radius = np.random.randint(img_array.shape[0]//10, img_array.shape[0]//5)
                intensity = np.random.uniform(0.7, 1.0)
                
                for i in range(img_array.shape[0]):
                    for j in range(img_array.shape[1]):
                        dist = np.sqrt((i - x)**2 + (j - y)**2)
                        if dist < radius:
                            heatmap[i, j] = max(heatmap[i, j], (1 - dist/radius) * intensity * score)
        
        # Ensure output directory exists
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating output directory: {str(e)}")
            
        # Normalize heatmap
        heatmap = np.clip(heatmap, 0, 1)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Convert RGB for display if needed
        if len(img_display.shape) == 2:  # If grayscale
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        elif img_display.shape[2] == 3:  # If RGB
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        # Overlay heatmap on image
        alpha = 0.4
        superimposed = cv2.addWeighted(
            img_display,
            1 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        # Add condition name and score
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            superimposed,
            f"{condition_name}: {score:.2f}",
            (10, 30),
            font, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            superimposed,
            "(Placeholder Explanation)",
            (10, superimposed.shape[0] - 10),
            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        # Save the visualization
        cv2.imwrite(output_path, superimposed)
        logger.info(f"Saved placeholder heatmap to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating placeholder heatmap: {str(e)}", exc_info=True)
        
        # Last-resort fallback - generate a simpler image
        try:
            # Convert to display format
            img_display = (img_array * 255).astype(np.uint8)
            if len(img_display.shape) == 2:  # If grayscale
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
            elif img_display.shape[2] == 3:  # If RGB
                img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
                
            # Add text directly
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img_display,
                f"{condition_name}: {score:.2f}",
                (10, 30),
                font, 0.7, (0, 0, 255), 2, cv2.LINE_AA
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save simple image
            cv2.imwrite(output_path, img_display)
            logger.info(f"Saved simplified fallback image to {output_path}")
            return True
            
        except Exception as nested_e:
            logger.error(f"Even fallback image generation failed: {str(nested_e)}")
            return False
