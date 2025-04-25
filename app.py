import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertModel
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import time

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize BERT
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.eval().to(device)

# Page configuration
st.set_page_config(
    page_title="Depression Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #e8f4f8;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        padding: 10px;
    }
    .st-bb {
        background-color: white;
    }
    .st-at {
        background-color: #e6f3ff;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Arial', sans-serif;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .form-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Depression level labels
levels = ['None', 'Mild', 'Moderate', 'Severe']

# Get absolute path to models
def get_model_path(model_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "processed_dataset")
    return os.path.join(processed_dir, f"{model_name}.keras")

# Load models with dynamic paths and error handling
@st.cache_resource
def load_models():
    models = {
        'autoencoder': None,
        'bert_only': None,
        'full_fusion': None
    }
    
    try:
        autoencoder_path = get_model_path("autoencoder")
        if os.path.exists(autoencoder_path):
            models['autoencoder'] = load_model(autoencoder_path)
    except Exception as e:
        st.error(f"Error loading autoencoder: {str(e)}")
    
    try:
        bert_only_path = get_model_path("bert_only_fusion_model")
        if os.path.exists(bert_only_path):
            models['bert_only'] = load_model(bert_only_path)
    except Exception as e:
        st.error(f"Error loading BERT-only model: {str(e)}")
    
    try:
        full_fusion_path = get_model_path("full_fusion_model")
        if os.path.exists(full_fusion_path):
            models['full_fusion'] = load_model(full_fusion_path)
    except Exception as e:
        st.error(f"Error loading full fusion model: {str(e)}")
    
    return models

models = load_models()

# Sample evaluation metrics (replace with your actual metrics)
def get_sample_metrics():
    return {
        'full_fusion': {
            'accuracy': 0.89,
            'f1': 0.87,
            'recall': 0.89,
            'confusion_matrix': np.array([[120, 5, 2, 1],
                                         [8, 110, 7, 3],
                                         [3, 6, 115, 4],
                                         [2, 3, 5, 118]]),
            'class_report': {
                '0': {'precision': 0.90, 'recall': 0.94, 'f1-score': 0.92, 'support': 128},
                '1': {'precision': 0.89, 'recall': 0.86, 'f1-score': 0.87, 'support': 128},
                '2': {'precision': 0.89, 'recall': 0.90, 'f1-score': 0.89, 'support': 128},
                '3': {'precision': 0.94, 'recall': 0.92, 'f1-score': 0.93, 'support': 128}
            }
        },
        'bert_only': {
            'accuracy': 0.82,
            'f1': 0.81,
            'recall': 0.82,
            'confusion_matrix': np.array([[110, 10, 5, 3],
                                        [12, 105, 8, 3],
                                        [7, 9, 105, 7],
                                        [5, 6, 10, 107]]),
            'class_report': {
                '0': {'precision': 0.82, 'recall': 0.86, 'f1-score': 0.84, 'support': 128},
                '1': {'precision': 0.81, 'recall': 0.82, 'f1-score': 0.81, 'support': 128},
                '2': {'precision': 0.82, 'recall': 0.82, 'f1-score': 0.82, 'support': 128},
                '3': {'precision': 0.89, 'recall': 0.84, 'f1-score': 0.86, 'support': 128}
            }
        }
    }

metrics = get_sample_metrics()

# Text preprocessing
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return ' '.join(tokens)

# BERT Embedding Function
def get_bert_embedding(text):
    try:
        if not text.strip():
            return np.zeros(768)
        
        inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    except Exception as e:
        st.error(f"Error generating BERT embedding: {str(e)}")
        return np.zeros(768)

# Chatbot responses
def get_chatbot_response(depression_level):
    responses = {
        0: [
            "It's great that you're feeling well! Remember to maintain healthy habits.",
            "You seem to be in a good place mentally. Keep up the positive outlook!",
            "Consider checking in with friends or family who might need support."
        ],
        1: [
            "You're showing some mild signs of stress. Consider relaxation techniques.",
            "Mild mood changes are common. Try getting some fresh air or exercise.",
            "Journaling might help you process what's on your mind."
        ],
        2: [
            "You're showing moderate signs of depression. Consider talking to someone.",
            "Moderate symptoms may benefit from professional support.",
            "Have you considered speaking with a counselor or therapist?"
        ],
        3: [
            "Please reach out to a mental health professional or crisis line immediately.",
            "Your responses indicate severe distress. You're not alone - help is available.",
            "Contact a trusted person or mental health service right away."
        ]
    }
    return np.random.choice(responses.get(depression_level, ["Thanks for sharing."]))

def get_severity_description(level):
    descriptions = [
        "Your responses indicate no significant signs of depression.",
        "Your responses suggest mild symptoms that may benefit from self-care strategies.",
        "Your responses indicate moderate symptoms that may benefit from professional support.",
        "Your responses suggest severe symptoms that warrant immediate professional attention."
    ]
    return descriptions[level]

def show_dashboard():
    st.title("Depression Detection System Dashboard")
    
    st.markdown("""
    ### Multi-Modal Depression Detection
    This system combines clinical data, physiological signals, social media analysis, and chatbot interactions
    to assess depression levels using advanced machine learning techniques.
    """)
    
    # Metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">'
                   '<h3>Full Fusion Model</h3>'
                   f'<p class="big-font">Accuracy: {metrics["full_fusion"]["accuracy"]*100:.1f}%</p>'
                   '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">'
                   '<h3>BERT-Only Model</h3>'
                   f'<p class="big-font">Accuracy: {metrics["bert_only"]["accuracy"]*100:.1f}%</p>'
                   '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">'
                   '<h3>System Coverage</h3>'
                   '<p class="big-font">4 Depression Levels</p>'
                   '</div>', unsafe_allow_html=True)
    
    # Depression level explanation
    st.markdown("""
    ### Depression Level Classification:
    - **0: None** - No significant signs of depression
    - **1: Mild** - Minor symptoms that may not interfere with daily life
    - **2: Moderate** - Noticeable symptoms that affect some daily activities
    - **3: Severe** - Significant impairment in daily functioning
    """)
    
    # Sample data visualization
    st.subheader("Sample Data Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = [128, 128, 128, 128]  # Sample balanced dataset
    sns.barplot(x=levels, y=counts, palette="Blues_d", ax=ax)
    ax.set_title("Distribution of Depression Levels in Training Data")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def show_model_performance():
    st.title("Model Performance Analysis")
    
    st.markdown("""
    ### Comparative Model Evaluation
    Below you can see the performance metrics for our two main models:
    - **Full Fusion Model**: Combines all data modalities (clinical, physiological, text)
    - **BERT-Only Model**: Uses only text data from social media and chatbot interactions
    """)
    
    # Model comparison - convert metrics to percentages
    comparison_data = {
        'Model': ['Full Fusion', 'BERT-Only'],
        'Accuracy': [metrics["full_fusion"]["accuracy"], metrics["bert_only"]["accuracy"]],
        'F1-Score': [metrics["full_fusion"]["f1"], metrics["bert_only"]["f1"]],
        'Recall': [metrics["full_fusion"]["recall"], metrics["bert_only"]["recall"]]
    }
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format as percentages without using style.format
    display_df = comparison_df.copy()
    for col in ['Accuracy', 'F1-Score', 'Recall']:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    # Display the formatted DataFrame
    st.dataframe(display_df.style.background_gradient(cmap='Blues'), 
                use_container_width=True)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Full Fusion Model**")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        sns.heatmap(metrics['full_fusion']['confusion_matrix'], annot=True, fmt='d',
                   cmap='Blues', xticklabels=levels, yticklabels=levels)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)
    
    with col2:
        st.markdown("**BERT-Only Model**")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        sns.heatmap(metrics['bert_only']['confusion_matrix'], annot=True, fmt='d',
                   cmap='Greens', xticklabels=levels, yticklabels=levels)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)
    
    # Classification reports
    st.subheader("Detailed Classification Reports")
    
    tab1, tab2 = st.tabs(["Full Fusion Model", "BERT-Only Model"])
    
    with tab1:
        st.markdown("**Full Fusion Model Classification Report**")
        report_df = pd.DataFrame(metrics['full_fusion']['class_report']).T
        # Convert to percentages and format
        report_df = report_df.applymap(lambda x: f"{x*100:.2f}%")
        st.dataframe(report_df.style.background_gradient(cmap='Blues'))
    
    with tab2:
        st.markdown("**BERT-Only Model Classification Report**")
        report_df = pd.DataFrame(metrics['bert_only']['class_report']).T
        # Convert to percentages and format
        report_df = report_df.applymap(lambda x: f"{x*100:.2f}%")
        st.dataframe(report_df.style.background_gradient(cmap='Greens'))

def show_chatbot():
    st.title("Depression Assessment Chatbot")
    
    st.markdown("""
    ### Interactive Depression Screening
    This chatbot analyzes your responses to assess potential depression symptoms.
    All conversations are anonymous and not stored.
    """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.conversation = []
        st.session_state.depression_level = None
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("How are you feeling today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation.append(("user", prompt))
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process message and generate response
        with st.spinner("Analyzing your response..."):
            try:
                # Preprocess text
                processed_text = preprocess_text(prompt)
                
                if not processed_text.strip():
                    response = "Could you please share more about how you're feeling?"
                else:
                    # Get BERT embedding
                    embedding = get_bert_embedding(processed_text)
                    
                    if models['bert_only'] is not None:
                        # Predict with BERT-only model
                        prediction = models['bert_only'].predict(
                            [embedding.reshape(1, -1), embedding.reshape(1, -1)], 
                            verbose=0
                        )
                        depression_level = np.argmax(prediction)
                        st.session_state.depression_level = depression_level
                        
                        # Get appropriate response
                        response = get_chatbot_response(depression_level)
                    else:
                        response = "Model not available. Please try again later."
            except Exception as e:
                st.error(f"Error processing your message: {str(e)}")
                response = "I encountered an error processing your message. Please try again."
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.conversation.append(("assistant", response))
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
    
    # Show depression level if detected
    if st.session_state.depression_level is not None:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Current Assessment</h3>
            <p class="big-font">Depression Level: {levels[st.session_state.depression_level]}</p>
            <p>{get_severity_description(st.session_state.depression_level)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Resources based on level
        if st.session_state.depression_level >= 2:
            st.warning("""
            **Recommended Resources:**
            - National Suicide Prevention Lifeline: 1-800-273-8255
            - Crisis Text Line: Text HOME to 741741
            - [Find a Therapist](https://www.psychologytoday.com/us/therapists)
            """)

def show_client_input():
    st.title("Client Depression Assessment")
    
    st.markdown("""
    ### Comprehensive Depression Screening
    Please fill out the following information to assess potential depression symptoms.
    """)
    
    with st.form("client_assessment"):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        
        # Personal Information
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=12, max_value=100, value=25)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
            email = st.text_input("Email (optional)")
        
        # Clinical Information
        st.subheader("Clinical Information")
        family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])
        treatment = st.selectbox("Currently in treatment?", ["No", "Yes"])
        work_interfere = st.selectbox("How often does mental health interfere with work?", 
                                    ["Never", "Rarely", "Sometimes", "Often"])
        
        # Physiological Information
        st.subheader("Physiological Information")
        col1, col2 = st.columns(2)
        with col1:
            sleep_duration = st.slider("Hours of sleep per night", 0.0, 12.0, 7.0, 0.5)
            heart_rate = st.number_input("Resting heart rate (bpm)", 40, 120, 72)
        with col2:
            sleep_quality = st.slider("Quality of sleep (1-10)", 1, 10, 7)
            daily_steps = st.number_input("Average daily steps", 0, 30000, 8000)
        
        # Free text input
        st.subheader("Additional Information")
        statement = st.text_area("How have you been feeling lately? (optional)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Submit Assessment")
        
        if submitted:
            with st.spinner("Analyzing your information..."):
                try:
                    # Prepare data for prediction
                    clinical_data = {
                        'Age': age,
                        'Gender': 1 if gender == "Male" else 0,
                        'family_history': 1 if family_history == "Yes" else 0,
                        'treatment': 1 if treatment == "Yes" else 0,
                        'work_interfere': work_interfere
                    }
                    
                    phys_data = {
                        'Sleep Duration': sleep_duration,
                        'Quality of Sleep': sleep_quality,
                        'Heart Rate': heart_rate,
                        'Daily Steps': daily_steps,
                        'Sleep_Quality_Index': sleep_duration * sleep_quality
                    }
                    
                    # Process text if provided
                    text_embedding = np.zeros(768)
                    if statement and isinstance(statement, str) and statement.strip():
                        processed_text = preprocess_text(statement)
                        text_embedding = get_bert_embedding(processed_text)
                    
                    # Make prediction if models are available
                    if models['full_fusion'] is not None:
                        # Prepare inputs for full fusion model
                        clinical_input = np.array([[clinical_data['Age'], 
                                                 clinical_data['Gender'], 
                                                 clinical_data['family_history'], 
                                                 1 if clinical_data['work_interfere'] in ["Often", "Sometimes"] else 0]]).astype('float32')
                        
                        phys_input = np.array([phys_data['Sleep Duration'],
                                             phys_data['Quality of Sleep'],
                                             phys_data['Heart Rate'],
                                             phys_data['Daily Steps'],
                                             phys_data['Sleep_Quality_Index']]).astype('float32')
                        
                        phys_cnn_input = phys_input.reshape(1, 5, 1)
                        phys_lstm_input = phys_input.reshape(1, 5, 1)
                        
                        text_input = text_embedding.reshape(1, -1)
                        
                        # Make prediction
                        prediction = models['full_fusion'].predict(
                            [clinical_input, phys_cnn_input, phys_lstm_input, text_input, text_input],
                            verbose=0
                        )
                        depression_level = np.argmax(prediction)
                        
                        # Show results
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Assessment Results</h3>
                            <p class="big-font">Depression Level: {levels[depression_level]}</p>
                            <p>{get_severity_description(depression_level)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Recommendations
                        if depression_level >= 2:
                            st.warning("""
                            **Recommended Next Steps:**
                            - Consider scheduling an appointment with a mental health professional
                            - Practice self-care techniques
                            - Reach out to trusted friends or family
                            """)
                        else:
                            st.success("""
                            **Maintenance Suggestions:**
                            - Continue healthy habits
                            - Monitor your mood regularly
                            - Don't hesitate to seek help if needed
                            """)
                    else:
                        st.error("Prediction model not available. Please try again later.")
                
                except Exception as e:
                    st.error(f"Error processing your assessment: {str(e)}")

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a section", 
                               ["Dashboard", "Model Performance", "Client Assessment", "Chatbot Interaction"])
    
    if app_mode == "Dashboard":
        show_dashboard()
    elif app_mode == "Model Performance":
        show_model_performance()
    elif app_mode == "Client Assessment":
        show_client_input()
    elif app_mode == "Chatbot Interaction":
        show_chatbot()

if __name__ == "__main__":
    main()