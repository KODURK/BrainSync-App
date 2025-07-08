import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys
import pandas as pd # For better table display in Streamlit

# --- Explicitly add the site-packages path to sys.path ---
# This is crucial for Streamlit to find packages in your activated conda environment.
# IMPORTANT: Replace this with YOUR actual path if it's different!
SITE_PACKAGES_PATH = r"C:\\Users\\SELASIE\\Desktop\\BrainTumour\\BrainTumorDetectorProject\\venv\\lib\\site-packages"
if SITE_PACKAGES_PATH not in sys.path:
    sys.path.append(SITE_PACKAGES_PATH)


# --- Configuration Parameters (MUST MATCH TRAINING) ---
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNELS = 3 # VGG16 expects 3 channels


# --- Paths to saved model and data files ---
MODEL_PATH = "brain_tumor_detection_model.h5"
TRAINING_HISTORY_PATH = "training_history.pkl"
CLASSIFICATION_REPORT_PATH = "classification_report.json"
CONFUSION_MATRIX_PATH = "confusion_matrix.json"


# --- Login Credentials (for demonstration purposes) ---
CORRECT_USERNAME = st.secrets.get("USERNAME", "admin")
CORRECT_PASSWORD = st.secrets.get("PASSWORD", "password123")


# --- Load Trained Model and Data (cached for efficiency) ---
@st.cache_resource
def load_brain_tumor_model():
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        st.warning("Please ensure the model file 'brain_tumor_detection_model.h5' is in the same directory as this app.py.")
        return None

@st.cache_data
def load_evaluation_data():
    """Loads classification report and confusion matrix from JSON files."""
    try:
        with open(CLASSIFICATION_REPORT_PATH, 'r') as f:
            report = json.load(f)
        with open(CONFUSION_MATRIX_PATH, 'r') as f:
            cm = np.array(json.load(f)) # Convert back to numpy array
        return report, cm
    except FileNotFoundError:
        st.error("Evaluation data files not found.")
        st.warning(f"Please ensure '{CLASSIFICATION_REPORT_PATH}' and '{CONFUSION_MATRIX_PATH}' "
                   "are in the main project directory. You need to run 'main_brain_tumor_app.py' first.")
        return None, None
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
        return None, None

@st.cache_data
def load_training_history():
    """Loads training history from a pickle file."""
    try:
        with open(TRAINING_HISTORY_PATH, 'rb') as f:
            history_data = pickle.load(f)
        # Ensure data is in Python list format for plotting if it was NumPy array in pickle
        for key in history_data:
            if isinstance(history_data[key], np.ndarray):
                history_data[key] = history_data[key].tolist()
        return history_data
    except FileNotFoundError:
        st.error("Training history file not found.")
        st.warning(f"Please ensure '{TRAINING_HISTORY_PATH}' is in the main project directory. You need to run 'main_brain_tumor_app.py' first.")
        return None
    except Exception as e:
        st.error(f"Error loading training history: {e}")
        return None


# Load all resources once at the start of the app
model_instance = load_brain_tumor_model()
classificationReportData, confusionMatrixData = load_evaluation_data()
trainingHistoryData = load_training_history()


# --- Utility Functions ---
def preprocess_image_for_streamlit_prediction(uploaded_file, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), channels=CHANNELS):
    """
    Loads and preprocesses an image uploaded via Streamlit for model prediction.
    Steps must match training preprocessing.
    """
    try:
        img = image.load_img(uploaded_file, target_size=target_size, color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, height, width, channels)
        img_array = img_array / 255.0 # Rescale pixels to [0, 1]
        return img_array
    except Exception as e:
        st.error(f"Error processing image for prediction: {e}")
        return None


def plot_chart(data, labels, title, x_label, y_label, y_min=None, y_max=None):
    """Generates a chart using matplotlib and displays it in Streamlit."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset in data:
        # Matplotlib compatible color codes (hex strings)
        ax.plot(labels, dataset['data'], label=dataset['label'], color=dataset['color'], alpha=dataset.get('alpha', 1), linestyle=dataset.get('linestyle', '-'))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory


# --- Page Content Functions ---

def home_page_content():
    st.markdown(
        """
        <h2 style='text-align: center; font-size: 2.25rem; font-weight: bold; color: #1a202c; margin-bottom: 2rem;'>BRAIN TUMOUR DETECTION</h2>
        <div style='max-width: 600px; margin: 0 auto; padding: 2rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <p style='font-size: 1.125rem; color: #4a5568; margin-bottom: 1.5rem; text-align: center;'>Upload Image:</p>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key="home_uploader")

    if uploaded_file is not None:
        col_img_l, col_img_c, col_img_r = st.columns([1, 2, 1])
        with col_img_c:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("")

    submit_button_key = "submit_prediction_button_" + str(hash(uploaded_file.name if uploaded_file else "no_file"))
    if st.button("Submit", key=submit_button_key):
        if model_instance is None:
            st.error("Model not loaded. Cannot make prediction.")
            return

        if uploaded_file is None:
            st.error("Please upload an image before clicking submit.")
            return

        with st.spinner("Analyzing image with your trained model..."):
            try:
                processed_img = preprocess_image_for_streamlit_prediction(uploaded_file)

                if processed_img is not None:
                    prediction_probability = model_instance.predict(processed_img)[0][0]
                    
                    if prediction_probability > 0.5:
                        result_label = "Brain Tumor Symptoms"
                        result_details = "unexplained weight loss, double vision, loss of vision, increased pressure felt in the back of the head, inability to speak, hearing loss, or numbness that gradually worsens on one side of the body."
                        prediction_class_style = "Tumor"
                    else:
                        result_label = "Normal"
                        result_details = ""
                        prediction_class_style = "Normal"

                    st.markdown(
                        f"""
                        <div style='border-left: 5px solid; padding: 1rem; margin-top: 1.5rem; border-radius: 0.5rem;"""
                        f""" border-color: {'#EF4444' if prediction_class_style == 'tumor' else '#10B981'};"""
                        f""" background-color: {'#FEF2F2' if prediction_class_style == 'tumor' else '#ECFDF5'};"""
                        f""" color: {'#B91C1C' if prediction_class_style == 'tumor' else '#047857'};'>"""
                        f"""
                            <p style='font-weight: bold; font-size: 1.25rem; margin-bottom: 0.5rem;'>Prediction: Result: <span id='prediction-label'>{result_label}</span></p>
                            <p style='font-size: 0.875rem; color: #4a5568;' id='prediction-details'>{result_details}</p>
                            <p style='font-size: 0.75rem; margin-top: 1rem; color: #6b7280; font-style: italic;'>
                                Disclaimer: This system provides a preliminary assessment based on its training. It is designed to assist medical professionals in research or screening. It is NOT a standalone diagnostic tool. Always consult a qualified healthcare professional for definitive diagnosis and treatment decisions.
                            </p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                else:
                    st.error("Failed to preprocess image for prediction.")

            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                st.warning("Please ensure the uploaded file is a valid image and the model loaded correctly.")
    
    st.markdown("</div>", unsafe_allow_html=True)


def login_page_content():
    st.markdown(
        """
        <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: calc(100vh - 10rem);'>
            <div style='max-width: 400px; width: 100%; background-color: white; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center;'>
                <h2 style='font-size: 2.25rem; font-weight: bold; color: #1a202c; margin-bottom: 1.5rem;'>LOGIN</h2>
        """,
        unsafe_allow_html=True
    )

    username = st.text_input("Username", key="login_username_input")
    password = st.text_input("Password", type="password", key="login_password_input")

    login_button_pressed = st.button("Login", key="login_submit_button", use_container_width=True)

    if login_button_pressed:
        if username == CORRECT_USERNAME and password == CORRECT_PASSWORD:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Logged in successfully!")
            st.rerun() # Rerun to refresh the entire app and show Home page
        else:
            st.error("Incorrect username or password.")
            
    st.markdown(
        """
            <p style='font-size: 0.875rem; color: #6b7280; margin-top: 1.5rem;'>Hint: Admin / Password123</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def preview_page_content():
    st.markdown(
        """
        <h2 style='text-align: center; font-size: 2.25rem; font-weight: bold; color: #1a202c; margin-bottom: 2rem;'>PROJECT OVERVIEW</h2>
        <div style='max-width: 900px; margin: 0 auto; padding: 2rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
            <p style='font-size: 1.125rem; color: #4a5568; margin-bottom: 1.5rem; text-align: center;'>
                This project aims to develop an intelligence system for the detection of brain tumors from MRI (Magnetic Resonance Imaging) scans. Leveraging the power of deep learning, this system is designed to assist medical professionals in preliminary screening and enhance diagnostic efficiency.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    # Replicated "Our Dedicated Team" section
    st.markdown("""
        <div style='max-width: 900px; margin: 3rem auto; padding: 2rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='font-size: 1.75rem; font-weight: bold; color: #1a202c; margin-bottom: 1.5rem; text-align: center;'>Our Dedicated Team</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;'>
                <!-- Supervisor -->
                <div style='background-color: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
                    <div style='width: 8rem; height: 8rem; margin: 0 auto 1rem; border-radius: 50%; overflow: hidden; border: 4px solid #8b5cf6; display: flex; align-items: center; justify-content: center; background-color: #c7d2fe;'>
                        <img src="https://placehold.co/128x128/9CA3AF/FFFFFF?text=Supervisor" alt="Supervisor" style="width: 100%; height: 100%; object-fit: cover;">
                    </div>
                    <h4 style='font-size: 1.125rem; font-weight: 600; color: #1a202c; margin-bottom: 0.5rem;'>DR. MOSES A. AGEBURE</h4>
                    <p style='font-size: 0.875rem; font-weight: 500; color: #4f46e5;'>Project Supervisor</p>
                    <p style='font-size: 0.875rem; color: #4a5568; margin-top: 0.75rem;'>Leading the vision and scientific direction of the Brain Tumour Detection project.</p>
                </div>
                <!-- Researcher -->
                <div style='background-color: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
                    <div style='width: 8rem; height: 8rem; margin: 0 auto 1rem; border-radius: 50%; overflow: hidden; border: 4px solid #8b5cf6; display: flex; align-items: center; justify-content: center; background-color: #c7d2fe;'>
                        <img src="https://placehold.co/128x128/9CA3AF/FFFFFF?text=Researcher" alt="Researcher" style="width: 100%; height: 100%; object-fit: cover;">
                    </div>
                    <h4 style='font-size: 1.125rem; font-weight: 600; color: #1a202c; margin-bottom: 0.5rem;'>KUMORDZI SELASI GILBERT</h4>
                    <p style='font-size: 0.875rem; font-weight: 500; color: #4f46e5;'>Lead Researcher</p>
                    <p style='font-size: 0.875rem; color: #4a5568; margin-top: 0.75rem;'>Specializing in AI model development and data analysis for medical imaging.</p>
                </div>
                <!-- Co-Researcher -->
                <div style='background-color: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
                    <div style='width: 8rem; height: 8rem; margin: 0 auto 1rem; border-radius: 50%; overflow: hidden; border: 4px solid #8b5cf6; display: flex; align-items: center; justify-content: center; background-color: #c7d2fe;'>
                        <img src="https://placehold.co/128x128/9CA3AF/FFFFFF?text=Co-Researcher" alt="Co-Researcher" style="width: 100%; height: 100%; object-fit: cover;">
                    </div>
                    <h4 style='font-size: 1.125rem; font-weight: 600; color: #1a202c; margin-bottom: 0.5rem;'>ZURI DONATUS YEMINOW</h4>
                    <p style='font-size: 0.875rem; font-weight: 500; color: #4f46e5;'>Co-Researcher</p>
                    <p style='font-size: 0.875rem; color: #4a5568; margin-top: 0.75rem;'>Contributing to algorithm optimization and clinical validation of the system.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    # Replicated "Contact Us" section
    st.markdown("""
        <div style='max-width: 800px; margin: 3rem auto; padding: 2rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='font-size: 1.75rem; font-weight: bold; color: #1a202c; margin-bottom: 1.5rem; text-align: center;'>Contact Us</h3>
            <p style='font-size: 1.125rem; color: #4a5568; margin-bottom: 2rem; text-align: center;'>
                Have questions or feedback? Feel free to reach out to us.
            </p>
            <div style='display: flex; justify-content: center; gap: 1.5rem;'>
                <a href='mailto:info@braindetection.com' style='display: flex; align-items: center; justify-content: center; background-color: #4f46e5; color: white; font-weight: bold; padding: 0.75rem 1.5rem; border-radius: 9999px; text-decoration: none; transition: background-color 0.2s;'>
                    <svg class='w-5 h-5 mr-2' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M3 8l7.89 5.26a2 2 0 002.22 0L21 8m-2 10a2 2 0 01-2 2H7a2 2 0 01-2-2V8a2 2 0 012-2h10a2 2 0 012 2v10z'></path></svg>
                    Email Us: gilbertselasi3@gmail.com / kingselasie80@gmail.com
                </a>
                <a href='tel:+1234567890' style='display: flex; align-items: center; justify-content: center; background-color: #e2e8f0; color: #4a5568; font-weight: bold; padding: 0.75rem 1.5rem; border-radius: 9999px; text-decoration: none; transition: background-color 0.2s;'>
                    <svg class='w-5 h-5 mr-2' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'><path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z'></path></svg>
                    Call Us: +233-54-499-4448
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center; font-size: 0.875rem; color: #6b7280; margin-top: 1rem;'>&copy; BrainSync Detect 2025. All rights reserved.</p>", unsafe_allow_html=True)


def performance_analysis_page_content():
    st.markdown("<h2 style='text-align: center; font-size: 2.25rem; font-weight: bold; color: #1a202c; margin-bottom: 2rem;'>PERFORMANCE ANALYSIS</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; font-size: 1.125rem; color: #4a5568; margin-bottom: 1.5rem;'>
            Key metrics demonstrating the overall performance of the trained model on its validation dataset.
            These metrics characterize the model's general capabilities, not the analysis of a single uploaded image.
        </p>
        """, unsafe_allow_html=True)

    if classificationReportData is None or confusionMatrixData is None:
        st.error("Evaluation data files not found or not loaded.")
        st.warning(f"Please ensure '{CLASSIFICATION_REPORT_PATH}' and '{CONFUSION_MATRIX_PATH}' "
                   "are in the main project directory. You need to run 'main_brain_tumor_app.py' first.")
        return

    col_acc, col_prec, col_rec = st.columns(3)
    with col_acc:
        st.markdown(f"<div style='background-color: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>"
                    f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #4f46e5;'>ACCURACY</h3>"
                    f"<p style='font-size: 2.25rem; font-weight: bold; color: #1a202c; margin-top: 0.5rem;'>{classificationReportData['accuracy']:.3f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col_prec:
        st.markdown(f"<div style='background-color: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>"
                    f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #4f46e5;'>PRECISION</h3>"
                    f"<p style='font-size: 2.25rem; font-weight: bold; color: #1a202c; margin-top: 0.5rem;'>{classificationReportData['yes']['precision']:.3f}</p>"
                    "</div>", unsafe_allow_html=True)
    with col_rec:
        st.markdown(f"<div style='background-color: #f8fafc; padding: 1.5rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>"
                    f"<h3 style='font-size: 1.25rem; font-weight: 600; color: #4f46e5;'>RECALL</h3>"
                    f"<p style='font-size: 2.25rem; font-weight: bold; color: #1a202c; margin-top: 0.5rem;'>{classificationReportData['yes']['recall']:.3f}</p>"
                    "</div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style='text-align: center; margin-top: 1.5rem;'>
            <p style='font-size: 1.5rem; font-weight: 600; color: #4f46e5;'>F-MEASURE: <span style='font-size: 2.25rem; font-weight: bold; color: #1a202c;'>{classificationReportData['yes']['f1-score']:.3f}</span></p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Confusion Matrix")
    st.write("This chart visualizes the model's correct and incorrect predictions on the **validation dataset**.")

    fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    labels_cm = ['True Negative (No Tumor)', 'False Positive (No -> Tumor)',
                 'False Negative (Tumor -> No)', 'True Positive (Tumor)']
    sizes_cm = [confusionMatrixData[0][0], confusionMatrixData[0][1],
                confusionMatrixData[1][0], confusionMatrixData[1][1]]
    colors_cm = ['#10B981', '#F59E0B', '#EF4444', '#3B82F6'] # Green, Amber, Red, Blue

    ax_cm.pie(sizes_cm, labels=labels_cm, autopct='%1.1f%%', startangle=90, colors=colors_cm, pctdistance=0.85)
    ax_cm.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    ax_cm.set_title('Confusion Matrix Distribution', fontsize=16)
    st.pyplot(fig_cm)
    plt.close(fig_cm) # Close the figure to free up memory

    st.markdown("""
        **Interpretation of Confusion Matrix:**
        - **True Negatives:** Correctly predicted 'no tumor'.
        - **False Positives:** Incorrectly predicted 'tumor' (a false alarm).
        - **False Negatives:** Incorrectly predicted 'no tumor' when it was a 'tumor' (a missed tumor - **critical**).
        - **True Positives:** Correctly predicted 'tumor'.
        """)
    st.markdown("---")
    st.info("""
        **Professional Insight:** For medical diagnosis, **high Recall (Sensitivity)** for the 'Yes' (Tumor) class is paramount to minimize **False Negatives** (missing actual tumors).
        While precision is also important, a trade-off is often made to prioritize Recall in screening systems.
        """)


def training_charts_page_content():
    st.markdown("<h2 style='text-align: center; font-size: 2.25rem; font-weight: bold; color: #1a202c; margin-bottom: 2rem;'>CHARTS</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; font-size: 1.125rem; color: #4a5568; margin-bottom: 1.5rem;'>
            These charts visualize how the model's accuracy and loss evolved during the <strong>Training Process</strong>.
            They provide insight into the model's learning behavior and generalization over epochs.
        </p>
        """, unsafe_allow_html=True)

    if trainingHistoryData is None or len(trainingHistoryData['accuracy']) == 0:
        st.error("Training history data not found or is empty.")
        st.warning("Please ensure 'training_history.pkl' is in the main project directory and contains valid data. You need to run 'main_brain_tumor_app.py' first.")
        return

    epochs = list(range(1, len(trainingHistoryData['accuracy']) + 1))

    st.subheader("Accuracy Over Epochs")
    plot_chart(
        [
            {'label': 'Training Accuracy', 'data': trainingHistoryData['accuracy'], 'color': '#4F46E5'}, # Indigo-600
            {'label': 'Validation Accuracy', 'data': trainingHistoryData['val_accuracy'], 'color': '#F97316'} # Orange-500
        ],
        epochs,
        'Combined Training and Validation Accuracy (with Fine-tuning)',
        'Epoch',
        'Accuracy',
        y_min=0, y_max=1
    )

    st.subheader("Loss Over Epochs")
    plot_chart(
        [
            {'label': 'Training Loss', 'data': trainingHistoryData['loss'], 'color': '#4F46E5'}, # Indigo-600
            {'label': 'Validation Loss', 'data': trainingHistoryData['val_loss'], 'color': '#F97316'} # Orange-500
        ],
        epochs,
        'Combined Training and Validation Loss (with Fine-tuning)',
        'Epoch',
        'Loss',
        y_min=0
    )

    st.markdown("---")
    st.markdown("""
        **Interpretation of Charts:**
        - **Accuracy:** Ideally, both training and validation accuracy should consistently increase and converge.
        - **Loss:** Both training and validation loss should consistently decrease.
        - The point where Early Stopping activated is the epoch where the validation loss was at its lowest, and weights were restored from that point.
        """)


# --- Main Application Logic ---
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed" # Start with sidebar collapsed for cleaner look
)

# Initialize session state for page and login if not present
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Login" # Start directly on the Login page


# --- Custom Header Bar and Navigation ---
# Using st.columns to create the header structure
header_col1, header_col2 = st.columns([1, 3]) # Adjust column ratios for title and nav

with header_col1:
    st.markdown("<h1 style='font-size: 1.5rem; font-weight: bold; color: #1a202c; margin: 0;'>Brain Tumour</h1>", unsafe_allow_html=True)

with header_col2:
    # Use st.radio for horizontal tab-like navigation
    # This is more native to Streamlit and should be stable.
    page_selection = st.radio(
        "Navigation",
        options=["Home", "Login", "Preview", "Performance_analysis", "Chart", "Logout"],
        index=["Home", "Login", "Preview", "Performance_analysis", "Chart", "Logout"].index(st.session_state["current_page"]) if st.session_state["current_page"] in ["Home", "Login", "Preview", "Performance_analysis", "Chart", "Logout"] else 0, # Default to Home if state is invalid
        horizontal=True,
        label_visibility="collapsed" # Hide the default label for a cleaner tab look
    )

    # Update session state based on selected tab
    if page_selection != st.session_state["current_page"]:
        st.session_state["current_page"] = page_selection
        if page_selection == "Logout":
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.info("You have been logged out.")
        st.rerun() # Rerun to display the new page content

st.markdown("---") # A horizontal line for visual separation after the header


# --- Display Page Content Based on Login State and Selected Tab ---
if st.session_state["logged_in"]:
    # Display content based on the selected tab
    if st.session_state["current_page"] == "Home":
        home_page_content()
    elif st.session_state["current_page"] == "Performance_analysis":
        performance_analysis_page_content()
    elif st.session_state["current_page"] == "Chart":
        training_charts_page_content()
    elif st.session_state["current_page"] == "Preview":
        home_page_content()
    elif st.session_state["current_page"] == "Login": # If user clicks Login when already logged in
        home_page_content() # Show home page and give a warning
        st.warning("You are already logged in.")
    elif st.session_state["current_page"] == "Logout": # If Logout was selected
        login_page_content() # Redirect to login page after logout processing
        
else:
    # If not logged in, only show the login page
    # If a different tab was selected while logged out, redirect to login
    if st.session_state["current_page"] != "Login":
         st.session_state["current_page"] = "Login" # Force current page to Login
         st.rerun() # Rerun to show login page

    login_page_content() # Always render login content when not logged in
