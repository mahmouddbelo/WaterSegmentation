import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template, send_from_directory
import tifffile
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to avoid threading issues
import matplotlib.pyplot as plt
import json

app = Flask(__name__)

# Define folders for uploads and results
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
JSON_FOLDER = "static/json"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["JSON_FOLDER"] = JSON_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

# Define custom metrics and losses
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + 1 - dice_coefficient(y_true, y_pred)

def iou_score(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred))
    union = tf.keras.backend.sum(tf.keras.backend.abs(y_true)) + tf.keras.backend.sum(tf.keras.backend.abs(y_pred))
    return (intersection + smooth) / (union + smooth)

# Load the model with custom objects
model = keras.models.load_model(
    'model/water_segmentation_model_best.keras',
    custom_objects={'dice_coefficient': dice_coefficient, 'bce_dice_loss': bce_dice_loss, 'iou_score': iou_score}
)

# Function to preprocess TIFF image
def preprocess_tif(image_path):
    """ Load, normalize, and preprocess a TIFF image for the model. """
    try:
        img = tifffile.imread(image_path)
        
        # Get the original shape for debug info
        original_shape = img.shape
        print(f"Original image shape: {original_shape}")
        
        # Create a 12-channel image as required by the model
        if len(img.shape) == 2:
            # If grayscale, repeat it 12 times
            img_12ch = np.stack([img] * 12, axis=-1)
        elif len(img.shape) == 3:
            if img.shape[2] == 12:
                # Already has 12 channels
                img_12ch = img
            elif img.shape[2] > 12:
                # Take only the first 12 channels if more
                img_12ch = img[:, :, :12]
            else:
                # If fewer than 12 channels, repeat the existing channels
                # First, extract available channels
                available_channels = img.shape[2]
                repeat_factor = int(np.ceil(12 / available_channels))
                repeated_channels = np.repeat(img, repeat_factor, axis=2)
                img_12ch = repeated_channels[:, :, :12]
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
            
        print(f"After channel adjustment: {img_12ch.shape}")

        # Normalize image
        img_12ch = img_12ch.astype(np.float32) / 255.0

        # Resize if needed (adjust to model input size)
        img_12ch = tf.image.resize(img_12ch, (256, 256))
        
        # Ensure exactly 12 channels after resize
        if img_12ch.shape[-1] != 12:
            raise ValueError(f"Failed to create 12-channel image, got {img_12ch.shape[-1]} channels")

        # Add batch dimension
        img_12ch = np.expand_dims(img_12ch, axis=0)
        
        print(f"Final preprocessed shape: {img_12ch.shape}")
        return img_12ch, img  # Return both the 12-channel and original image
    except Exception as e:
        raise ValueError(f"Error processing TIFF image: {str(e)}")

# Function to segment image
def segment_image(image_path, output_path, json_path):
    """ Perform segmentation on the image using the model and save results. """
    try:
        print(f"Processing image: {image_path}")
        img_12ch, original_img = preprocess_tif(image_path)
        
        # Model prediction
        predictions = model.predict(img_12ch)
        print(f"Predictions shape: {predictions.shape}")

        # Convert to binary mask
        segmented_img = (predictions > 0.5).astype(np.uint8)
        
        # Prepare RGB version of original image for visualization
        if len(original_img.shape) == 2:
            original_img_rgb = np.stack([original_img] * 3, axis=-1)
        elif len(original_img.shape) == 3:
            if original_img.shape[2] >= 3:
                original_img_rgb = original_img[:, :, :3]  # Take first 3 channels for RGB
            else:
                # If fewer than 3 channels, repeat to get 3
                original_img_rgb = np.repeat(original_img, 3 // original_img.shape[2] + 1, axis=2)[:, :, :3]
        
        # Resize for consistent visualization
        original_img_resized = tf.image.resize(original_img_rgb, (256, 256)).numpy().astype(np.uint8)
        
        # Save mask as JSON for JavaScript visualization
        mask_binary = np.squeeze(segmented_img).astype(bool).tolist()
        with open(json_path, 'w') as f:
            json.dump({"mask": mask_binary}, f)

        # Save visualization as PNG
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original_img_resized)
        plt.title("Original Image (RGB Preview)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(segmented_img), cmap="Blues")
        plt.title("Segmented Mask")
        plt.axis("off")

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"Segmentation result saved to: {output_path}")
        print(f"Mask data saved to: {json_path}")
        return output_path
    except Exception as e:
        print(f"Error in segment_image: {str(e)}")
        # Save error info for debugging
        with open(f"{output_path}.error.txt", 'w') as f:
            f.write(f"Error: {str(e)}")
        return None

# Route for homepage (renders index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'message': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    try:
        # Save file to upload directory
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Generate result image path (save as .png)
        result_filename = f"seg_{os.path.splitext(file.filename)[0]}.png"
        result_image_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)
        
        # Generate JSON mask path
        json_filename = f"mask_{os.path.splitext(file.filename)[0]}.json"
        json_path = os.path.join(app.config["JSON_FOLDER"], json_filename)

        # Process the image and generate visualization
        segment_image(file_path, result_image_path, json_path)

        # Read original tiff image for RGB display
        try:
            original_img = tifffile.imread(file_path)
            
            # Convert to RGB for display
            if len(original_img.shape) == 2:
                # If grayscale, convert to RGB
                original_img_rgb = np.stack([original_img] * 3, axis=-1)
            elif len(original_img.shape) == 3:
                if original_img.shape[2] >= 3:
                    # Take first 3 channels for RGB
                    original_img_rgb = original_img[:, :, :3]
                else:
                    # If fewer than 3 channels, repeat to get 3
                    original_img_rgb = np.repeat(original_img, 3 // original_img.shape[2] + 1, axis=2)[:, :, :3]
            else:
                # Unexpected shape
                raise ValueError(f"Unexpected image shape: {original_img.shape}")
                
            # Clip values to valid range and convert to uint8
            original_img_rgb = np.clip(original_img_rgb, 0, 255).astype(np.uint8)
            
            # Resize for display
            original_img_resized = tf.image.resize(original_img_rgb, (256, 256)).numpy().astype(np.uint8)
            
            # Save as PNG for display
            original_filename = f"orig_{os.path.splitext(file.filename)[0]}.png"
            original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
            plt.imsave(original_path, original_img_resized)
        except Exception as e:
            print(f"Error saving RGB preview: {e}")
            # Create a simple error indicator image
            original_filename = f"orig_{os.path.splitext(file.filename)[0]}.png"
            original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
            error_img = np.zeros((256, 256, 3), dtype=np.uint8)
            error_img[:, :, 0] = 255  # Red image for error
            plt.imsave(original_path, error_img)

        return jsonify({
            'image_url': f"/static/results/{result_filename}",
            'original_url': f"/static/uploads/{original_filename}",
            'mask_url': f"/static/json/{json_filename}"
        }), 200

    except Exception as e:
        print(f"Error in upload_image: {e}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during processing. Check server logs for details.'
        }), 500

# Route to serve the segmentation result image
@app.route('/static/results/<filename>')
def get_result_image(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)

# Route to serve the JSON mask data
@app.route('/static/json/<filename>')
def get_json_data(filename):
    return send_from_directory(app.config["JSON_FOLDER"], filename)

# Route to serve the original image
@app.route('/static/uploads/<filename>')
def get_original_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == '__main__':
    app.run(debug=True)