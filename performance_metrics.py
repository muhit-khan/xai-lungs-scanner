import pandas as pd
import numpy as np
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import os
import time
import concurrent.futures
from io import BytesIO
from fpdf import FPDF

# --- Configuration ---
BASE_DIR = "NIH_Balanced_&_Resized_Chest_X-rays"
CSV_PATH = os.path.join(BASE_DIR, "new_labels.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "resized_images")
SAMPLE_SIZE = 10000  # Changed from 10 to 1000
OUTPUT_SAMPLE_CSV = "test_sample.csv"
API_URL = "https://rrdj617r-5000.asse.devtunnels.ms/predict"
IMAGE_SIZE = (224, 224)
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60
RETRY_DELAY_SECONDS = 5
OUTPUT_DIR = "performance_results"
API_KEY = "test_key"
LABEL_COLUMNS = []
WARNED_MISSING_API_LABELS = set()
MAX_WORKERS = 10  # Maximum number of parallel workers

# --- Helper Functions ---

def ensure_output_dir():
    """Ensures that the output directory and necessary subdirectories exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(os.path.join(OUTPUT_DIR, "confusion_matrices")):
        os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrices"))
    if not os.path.exists(os.path.join(OUTPUT_DIR, "roc_curves")):
        os.makedirs(os.path.join(OUTPUT_DIR, "roc_curves"))

def load_and_sample_data(csv_path, sample_size, output_csv_path):
    """Loads data from CSV, samples it, saves image identifiers of the sample, and identifies label columns."""
    global LABEL_COLUMNS
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Determine the image identifier column. Prioritize 'Path', then 'Image Index', then the first column.
    if 'Path' in df.columns:
        image_id_col = 'Path'
    elif 'Image Index' in df.columns:
        image_id_col = 'Image Index'
    elif df.columns.size > 0:
        image_id_col = df.columns[0]
        print(f"Warning: Neither 'Path' nor 'Image Index' column found. Using first column '{image_id_col}' as image identifier.")
    else:
        raise ValueError("CSV file has no columns.")

    # Get all label columns except 'No Finding' which is a special case and not in API response
    LABEL_COLUMNS = [col for col in df.columns if col != image_id_col and col != 'No Finding']
    if not LABEL_COLUMNS:
        raise ValueError("No label columns found in CSV. Ensure CSV has an image identifier column and label columns.")
    print(f"Identified image identifier column: '{image_id_col}'")
    print(f"Identified label columns: {LABEL_COLUMNS}")
    print(f"Note: 'No Finding' column excluded from analysis as it doesn't appear in API responses")

    # Log example API response structure for reference
    print("\nExample API Response Structure:")
    print("""
    {
        "explanations": { ... },
        "note": null,
        "predictions": [
            {
                "condition": "Atelectasis",
                "positive": true,
                "score": 0.5374575257301331
            },
            ...
        ],
        "success": true,
        "timestamp": "2025-05-14T11:29:51.735742"
    }
    """)

    if len(df) < sample_size:
        print(f"Warning: Requested sample size ({sample_size}) is larger than dataset size ({len(df)}). Using all available data.")
        sampled_df = df
    else:
        sampled_df = df.sample(n=sample_size, random_state=42) # random_state for reproducibility
    
    try:
        sampled_df[[image_id_col]].to_csv(output_csv_path, index=False)
        print(f"Sampled {len(sampled_df)} rows and saved image identifiers to {output_csv_path}")
    except OSError as e:
        print(f"Warning: Could not save sampled image identifiers to {output_csv_path} due to an OSError: {e}")
        print("Proceeding with in-memory sampled data. Please check available disk space if this file is required.")
    return sampled_df, image_id_col

def preprocess_image(image_path, target_size):
    """Loads, resizes an image, and returns it as a BytesIO buffer (PNG format)."""
    try:
        img = Image.open(image_path).convert('RGB') # Ensure 3 channels
        img = img.resize(target_size)
        
        # The prompt mentions "normalize". If the API expects a raw image file,
        # normalization is typically handled by the server-side model.
        # Here, we are sending the image file data.
        # If client-side normalization into a specific array format is needed,
        # this function and the API call would need to change significantly.
        
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_with_retry(api_url, image_buffer, image_filename, max_retries=MAX_RETRIES, timeout=TIMEOUT_SECONDS, delay=RETRY_DELAY_SECONDS):
    """Sends a POST request with the image to the API, with retries and timeout.
    Parses the new API response structure to extract predictions and probabilities
    ordered according to global LABEL_COLUMNS.
    """
    global WARNED_MISSING_API_LABELS
    # Aligning with Postman: 'file' is the field name for the uploaded file.
    files = {'file': (image_filename, image_buffer, 'image/png')}
    headers = {
        'X-API-Key': API_KEY
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, files=files, headers=headers, timeout=timeout)
            response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
            
            response_data = response.json()

            if not response_data.get("success"):
                print(f"API call for {image_filename} was not successful according to response: {response_data.get('note', 'No note provided.')}")
                return None

            api_predictions_list = response_data.get("predictions")
            if not isinstance(api_predictions_list, list):
                print(f"API response for {image_filename} is missing 'predictions' list or it's not a list.")
                return None

            # Create a dictionary for quick lookup of API predictions by condition name
            api_preds_map = {}
            for item in api_predictions_list:
                if isinstance(item, dict) and "condition" in item and "positive" in item and "score" in item:
                    api_preds_map[item["condition"]] = {
                        "positive": item["positive"],
                        "score": item["score"]
                    }
                else:
                    print(f"Warning: Malformed prediction item in API response for {image_filename}: {item}")
            
            ordered_predictions = []
            ordered_probabilities = []

            if not LABEL_COLUMNS: 
                print(f"Error: LABEL_COLUMNS is empty. Cannot order API predictions for {image_filename}.")
                # This is a critical configuration error, should ideally not happen if load_and_sample_data ran.
                return None 

            for label_name in LABEL_COLUMNS:
                if label_name in api_preds_map:
                    prediction_item = api_preds_map[label_name]
                    ordered_predictions.append(1 if prediction_item["positive"] else 0)
                    ordered_probabilities.append(float(prediction_item["score"]))
                else:
                    if label_name not in WARNED_MISSING_API_LABELS:
                        print(f"Warning: Label '{label_name}' not found in API response predictions (e.g., for {image_filename}). Defaulting to (0, 0.5). This warning will not repeat for this specific label.")
                        WARNED_MISSING_API_LABELS.add(label_name)
                    ordered_predictions.append(0) 
                    ordered_probabilities.append(0.5)

            return {"predictions": ordered_predictions, "probabilities": ordered_probabilities}

        except requests.exceptions.Timeout:
            print(f"Attempt {attempt + 1}/{max_retries}: API request timed out for {image_filename}.")
        except requests.exceptions.ConnectionError:
            print(f"Attempt {attempt + 1}/{max_retries}: API connection error for {image_filename}.")
        except requests.exceptions.HTTPError as e:
            print(f"Attempt {attempt + 1}/{max_retries}: HTTP error {e.response.status_code} for {image_filename} - {e.response.text}")
        except requests.exceptions.JSONDecodeError:
            print(f"Attempt {attempt + 1}/{max_retries}: Failed to decode JSON response for {image_filename}.")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries}: API request failed for {image_filename}: {e}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"Max retries reached for {image_filename}. API call failed.")
            return None

def calculate_and_display_metrics(y_true, y_pred, label_names):
    """Calculates and displays performance metrics."""
    # Overall subset accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Micro-averaged metrics (across all samples and classes)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Macro-averaged metrics (unweighted mean of per-class metrics)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate accuracy per class
    accuracy_per_class = []
    for i in range(y_true.shape[1]):
        accuracy_per_class.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    
    # Summary metrics
    metrics_summary = {
        "Overall Accuracy (Subset)": accuracy,
        "Precision (micro)": precision_micro,
        "Recall (micro)": recall_micro,
        "F1 Score (micro)": f1_micro,
        "Precision (macro)": precision_macro,
        "Recall (macro)": recall_macro,
        "F1 Score (macro)": f1_macro
    }
    
    per_class_metrics = {
        "Accuracy": accuracy_per_class,
        "Precision": precision_per_class,
        "Recall": recall_per_class,
        "F1 Score": f1_per_class
    }
    
    print("\n--- Performance Metrics ---")
    for name, value in metrics_summary.items():
        print(f"{name}: {value:.2f}")
    
    print("\n--- Per-Class Metrics ---")
    for i, label in enumerate(label_names):
        print(f"{label}:")
        for metric_name, metric_values in per_class_metrics.items():
            print(f"  {metric_name}: {metric_values[i]:.2f}")
    
    return metrics_summary, per_class_metrics

def generate_visualizations(y_true, y_pred, y_prob, metrics_summary, per_class_metrics, label_names, output_dir):
    """Generates and saves visualizations: confusion matrices, metrics bar plot, ROC curves, and per-class metrics."""
    # 1. Confusion Matrix Heatmaps (per label)
    if y_true.size == 0 or y_pred.size == 0:
        print("Not enough data to generate confusion matrices.")
    else:
        try:
            mcm = multilabel_confusion_matrix(y_true, y_pred)
            for i, label_name in enumerate(label_names):
                plt.figure(figsize=(6, 4))
                sns.heatmap(mcm[i], annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Predicted Negative', 'Predicted Positive'],
                            yticklabels=['Actual Negative', 'Actual Positive'])
                plt.title(f'Confusion Matrix for {label_name}')
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                cm_filename = "".join(c if c.isalnum() else "_" for c in label_name) # Sanitize filename
                cm_path = os.path.join(output_dir, "confusion_matrices", f"cm_{cm_filename}.png")
                plt.savefig(cm_path)
                plt.close()
                print(f"Saved confusion matrix for {label_name} to {cm_path}")
        except Exception as e:
            print(f"Error generating confusion matrices: {e}")


    # 2. Overall Metrics Bar Plot
    plt.figure(figsize=(10, 6))
    metrics_names = list(metrics_summary.keys())
    metrics_values = [round(v, 2) for v in metrics_summary.values()]
    bars = plt.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.ylabel("Score")
    plt.title("Overall Performance Metrics (micro-averaged)")
    plt.ylim(0, max(1.0, max(metrics_values) * 1.1 if metrics_values else 1.0) ) # Adjust ylim
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    barplot_path = os.path.join(output_dir, "metrics_barplot.png")
    plt.savefig(barplot_path)
    plt.close()
    print(f"Saved metrics bar plot to {barplot_path}")

    # 3. Per-class metrics bar charts
    for metric_name, metric_values in per_class_metrics.items():
        plt.figure(figsize=(15, 8))
        bars = plt.bar(label_names, metric_values, color='skyblue')
        plt.ylabel(metric_name)
        plt.xlabel("Condition")
        plt.title(f"{metric_name} Across All Conditions")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"{metric_name.lower().replace(' ', '_')}_bar_chart.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"Saved {metric_name} bar chart to {chart_path}")

    # 4. ROC Curve (per label and micro-averaged, if probabilities available)
    if y_prob is not None and y_prob.size > 0 and y_prob.shape == y_true.shape:
        # Per-label ROC
        for i, label_name in enumerate(label_names):
            if np.unique(y_true[:, i]).size < 2: # Skip if only one class present in true labels for this category
                print(f"Skipping ROC for {label_name} as it has only one class in y_true.")
                continue
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, lw=2, label=f'{label_name} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {label_name}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_filename = "".join(c if c.isalnum() else "_" for c in label_name) # Sanitize filename
            roc_path = os.path.join(output_dir, "roc_curves", f"roc_{roc_filename}.png")
            plt.savefig(roc_path)
            plt.close()
            print(f"Saved ROC curve for {label_name} to {roc_path}")

        # Micro-averaged ROC
        if y_true.size > 0 and y_prob.size > 0 : # Ensure there's data
            try:
                fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_prob.ravel())
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                plt.figure(figsize=(8,6))
                plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC (AUC = {roc_auc_micro:.2f})', color='deeppink', linestyle=':', linewidth=4)
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Micro-averaged ROC Curve')
                plt.legend(loc="lower right")
                plt.tight_layout()
                micro_roc_path = os.path.join(output_dir, "roc_micro_average.png")
                plt.savefig(micro_roc_path)
                plt.close()
                print(f"Saved micro-averaged ROC curve to {micro_roc_path}")
            except Exception as e:
                 print(f"Error generating micro-averaged ROC curve: {e}")
    else:
        print("Probability scores not available, inconsistent, or y_true empty/malformed; skipping ROC curve generation.")

def save_metrics_to_csv(metrics_summary, per_class_metrics, label_names, output_dir):
    """Saves the metrics to a CSV file."""
    # Create a DataFrame for overall metrics
    overall_df = pd.DataFrame({"Metric": list(metrics_summary.keys()), 
                               "Value": list(metrics_summary.values())})
    
    # Create a DataFrame for per-class metrics
    per_class_data = []
    for i, label in enumerate(label_names):
        for metric_name, metric_values in per_class_metrics.items():
            per_class_data.append({
                "Label": label,
                "Metric": metric_name,
                "Value": metric_values[i]
            })
    per_class_df = pd.DataFrame(per_class_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "performance_report.csv")
    
    with open(csv_path, 'w', newline='') as f:
        f.write("OVERALL METRICS\n")
        overall_df.to_csv(f, index=False)
        f.write("\nPER-CLASS METRICS\n")
        per_class_df.to_csv(f, index=False)
    
    print(f"Saved metrics to {csv_path}")

def save_report_as_pdf(metrics_summary, per_class_metrics, label_names, output_dir):
    """Saves the metrics summary to a PDF file."""
    pdf = FPDF(format='A4')
    pdf.add_page()
    
    # Use the newer FPDF API with 'text' parameter instead of deprecated 'txt'
    # And use supported font
    pdf.set_font("Helvetica", 'B', size=16)
    pdf.cell(0, 10, text="Performance Metrics Report", align="C")
    pdf.ln(15)
    
    # Overall metrics
    pdf.set_font("Helvetica", 'B', size=12)
    pdf.cell(0, 10, text="Overall Metrics:")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=12)
    for metric, value in metrics_summary.items():
        pdf.cell(0, 10, text=f"{metric}: {value:.2f}")
        pdf.ln(8)
    
    # Per-class metrics
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', size=12)
    pdf.cell(0, 10, text="Per-Class Metrics:")
    pdf.ln(10)
    
    for i, label in enumerate(label_names):
        pdf.set_font("Helvetica", 'B', size=11)
        pdf.cell(0, 10, text=f"{label}:")
        pdf.ln(8)
        
        pdf.set_font("Helvetica", size=10)
        for metric_name, metric_values in per_class_metrics.items():
            pdf.cell(0, 8, text=f"  {metric_name}: {metric_values[i]:.2f}")
            pdf.ln(6)
        pdf.ln(2)
    
    # Visualization info - using shorter paths to avoid horizontal space issues
    pdf.ln(5)
    pdf.set_font("Helvetica", 'B', size=12)
    pdf.cell(0, 10, text="Visualizations Saved:")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=10)
    
    # Break long paths into multiple cells to prevent overflow
    pdf.cell(0, 8, text="- Confusion matrices:")
    pdf.ln(6)
    pdf.cell(5)  # indent 
    pdf.cell(0, 8, text=f"  {os.path.join(output_dir, 'confusion_matrices')}")
    pdf.ln(8)
    
    pdf.cell(0, 8, text="- Overall metrics bar plot:")
    pdf.ln(6)
    pdf.cell(5)  # indent
    pdf.cell(0, 8, text=f"  {os.path.join(output_dir, 'metrics_barplot.png')}")
    pdf.ln(8)
    
    pdf.cell(0, 8, text="- Per-class metrics bar charts:")
    pdf.ln(6)
    pdf.cell(5)  # indent
    pdf.cell(0, 8, text=f"  {output_dir}")
    pdf.ln(8)
    
    pdf.cell(0, 8, text="- ROC curves:")
    pdf.ln(6)
    pdf.cell(5)  # indent
    pdf.cell(0, 8, text=f"  {os.path.join(output_dir, 'roc_curves')}")
    pdf.ln(8)
    
    pdf.cell(0, 8, text="- CSV report:")
    pdf.ln(6)
    pdf.cell(5)  # indent
    pdf.cell(0, 8, text=f"  {os.path.join(output_dir, 'performance_report.csv')}")
    pdf.ln(8)

    pdf_path = os.path.join(output_dir, "performance_report.pdf")
    try:
        pdf.output(pdf_path)
        print(f"Saved metrics report to {pdf_path}")
    except Exception as e:
        print(f"Error saving PDF report: {e}")
        print("Consider shortening output directory path if the error persists.")

def process_image(row_tuple, image_id_col):
    """Processes a single image and returns the results. Used for parallel processing."""
    row_idx, row = row_tuple
    image_filename = row[image_id_col]
    image_path = os.path.join(IMAGES_DIR, str(image_filename))
    
    try:
        true_labels = [int(row[label]) for label in LABEL_COLUMNS]
    except KeyError as e:
        print(f"Skipping {image_filename}: Label column {e} not found in the row.")
        return None
    except ValueError as e:
        print(f"Skipping {image_filename}: Non-integer value for label ({e}).")
        return None

    image_buffer = preprocess_image(image_path, IMAGE_SIZE)
    if image_buffer is None:
        return None 

    api_response = predict_with_retry(API_URL, image_buffer, image_filename)
    image_buffer.close()

    if api_response:
        pred_labels_api = api_response["predictions"]
        pred_probs_api = api_response["probabilities"]
        
        try:
            pred_labels = [int(p) for p in pred_labels_api]
            pred_probs = [float(p) for p in pred_probs_api]
            
            return {
                "true_labels": true_labels,
                "pred_labels": pred_labels,
                "pred_probs": pred_probs
            }

        except ValueError as e:
            print(f"Warning: Error converting predictions/probabilities for {image_filename} to final types: {e}.")
            return None
    
    return None

# --- Main Script ---
def main():
    ensure_output_dir()
    
    try:
        sampled_df, image_id_col = load_and_sample_data(CSV_PATH, SAMPLE_SIZE, OUTPUT_SAMPLE_CSV)
    except FileNotFoundError as e:
        print(f"Error: {e}. Exiting.")
        return
    except ValueError as e:
        print(f"Error loading/sampling data: {e}. Exiting.")
        return

    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = [] 

    if not LABEL_COLUMNS:
        print("Error: Label columns were not identified from CSV. Cannot proceed.")
        return

    print(f"\nProcessing {len(sampled_df)} sampled images using API: {API_URL}")
    
    # Parallelize image processing with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list of futures
        futures = [executor.submit(process_image, (idx, row), image_id_col) 
                   for idx, row in sampled_df.iterrows()]
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Processing images"):
            result = future.result()
            if result:
                all_true_labels.append(result["true_labels"])
                all_pred_labels.append(result["pred_labels"])
                all_pred_probs.append(result["pred_probs"])

    if not all_true_labels or not all_pred_labels:
        print("No valid predictions were processed. Cannot calculate metrics.")
        return

    y_true_np = np.array(all_true_labels)
    y_pred_np = np.array(all_pred_labels)
    
    y_prob_np = None
    if all_pred_probs and len(all_pred_probs) == len(y_true_np):
        try:
            y_prob_np = np.array(all_pred_probs)
            if y_prob_np.shape != y_true_np.shape:
                print(f"Warning: Shape mismatch for probabilities array {y_prob_np.shape} vs true labels {y_true_np.shape}. ROC curves may be affected.")
                y_prob_np = None 
        except Exception as e:
            print(f"Could not convert probabilities to numpy array: {e}. ROC curves will be skipped or affected.")
            y_prob_np = None
    else:
        print("Probabilities list empty or length mismatch with true labels. ROC curves will be skipped or affected.")

    metrics_summary, per_class_metrics = calculate_and_display_metrics(y_true_np, y_pred_np, LABEL_COLUMNS)
    generate_visualizations(y_true_np, y_pred_np, y_prob_np, metrics_summary, per_class_metrics, LABEL_COLUMNS, OUTPUT_DIR)
    save_metrics_to_csv(metrics_summary, per_class_metrics, LABEL_COLUMNS, OUTPUT_DIR)
    save_report_as_pdf(metrics_summary, per_class_metrics, LABEL_COLUMNS, OUTPUT_DIR)

    print(f"\nPerformance evaluation complete. Processed {len(y_true_np)} images out of {SAMPLE_SIZE} sampled.")
    print(f"Results, plots, CSV report, and PDF summary saved in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()
