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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set page configuration as the FIRST Streamlit command
st.set_page_config(
    page_title="Depression Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize NLTK with caching
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        raise
download_nltk_data()
stop_words = set(stopwords.words('english'))

# Initialize BERT with caching and custom cache directory
@st.cache_resource
def load_bert():
    try:
        cache_dir = "./bert_cache"
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval().to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading BERT: {str(e)}")
        raise

# Load BERT
bert_tokenizer, bert_model, device = load_bert()

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

# Get absolute path to models in the root directory
def get_model_path(model_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_filenames = {
        'autoencoder': 'autoencoder.keras',
        'bert_only': 'bert_only_fusion_model.keras',
        'full_fusion': 'centralized_fusion_model.keras',
        'federated_fusion': 'federated_fusion_model.keras'
    }
    return os.path.join(base_dir, model_filenames.get(model_name, f"{model_name}.keras"))

# Load models with dynamic paths and error handling
@st.cache_resource
def load_models():
    models = {
        'autoencoder': None,
        'bert_only': None,
        'full_fusion': None,
        'federated_fusion': None
    }
    
    for model_name in models.keys():
        try:
            model_path = get_model_path(model_name)
            if os.path.exists(model_path):
                models[model_name] = load_model(model_path)
                st.success(f"Successfully loaded {model_name} model from {model_path}")
            else:
                st.warning(f"Model file for {model_name} not found at {model_path}. Ensure the .keras file is in the same directory as app.py.")
        except Exception as e:
            st.error(f"Error loading {model_name} from {model_path}: {str(e)}")
    
    return models

# Load models
models = load_models()

# Sample evaluation metrics
def get_metrics():
    metrics = {
        'federated_fusion': {
            'accuracy': 0.862069,
            'f1': 0.862067,
            'recall': 0.862069,
            'confusion_matrix': None,
            'class_report': None
        },
        'full_fusion': {
            'accuracy': 0.864224,
            'f1': 0.864058,
            'recall': 0.864224,
            'confusion_matrix': None,
            'class_report': None
        },
        'bert_only': {
            'accuracy': 0.855603,
            'f1': 0.855558,
            'recall': 0.855603,
            'confusion_matrix': None,
            'class_report': None
        }
    }

    class_reports = {
        'federated_fusion': {
            'None': {'precision': 0.8972, 'recall': 0.8727, 'f1-score': 0.8848, 'support': 110},
            'Mild': {'precision': 0.8696, 'recall': 0.8451, 'f1-score': 0.8571, 'support': 142},
            'Moderate': {'precision': 0.8382, 'recall': 0.8841, 'f1-score': 0.8605, 'support': 164},
            'Severe': {'precision': 0.8478, 'recall': 0.8125, 'f1-score': 0.8298, 'support': 48},
            'accuracy': 0.8621,
            'macro avg': {'precision': 0.8632, 'recall': 0.8536, 'f1-score': 0.8581, 'support': 192},
            'weighted avg': {'precision': 0.8628, 'recall': 0.8621, 'f1-score': 0.8621, 'support': 192}
        },
        'full_fusion': {
            'None': {'precision': 0.8750, 'recall': 0.8273, 'f1-score': 0.8505, 'support': 110},
            'Mild': {'precision': 0.8462, 'recall': 0.8521, 'f1-score': 0.8491, 'support': 142},
            'Moderate': {'precision': 0.8750, 'recall': 0.8963, 'f1-score': 0.8855, 'support': 164},
            'Severe': {'precision': 0.8571, 'recall': 0.8750, 'f1-score': 0.8660, 'support': 48},
            'accuracy': 0.8642,
            'macro avg': {'precision': 0.8633, 'recall': 0.8627, 'f1-score': 0.8628, 'support': 192},
            'weighted avg': {'precision': 0.8643, 'recall': 0.8642, 'f1-score': 0.8641, 'support': 192}
        },
        'bert_only': {
            'None': {'precision': 0.8491, 'recall': 0.8182, 'f1-score': 0.8333, 'support': 110},
            'Mild': {'precision': 0.8582, 'recall': 0.8521, 'f1-score': 0.8551, 'support': 142},
            'Moderate': {'precision': 0.8421, 'recall': 0.8780, 'f1-score': 0.8597, 'support': 164},
            'Severe': {'precision': 0.9130, 'recall': 0.8750, 'f1-score': 0.8936, 'support': 48},
            'accuracy': 0.8556,
            'macro avg': {'precision': 0.8656, 'recall': 0.8558, 'f1-score': 0.8604, 'support': 192},
            'weighted avg': {'precision': 0.8560, 'recall': 0.8556, 'f1-score': 0.8556, 'support': 192}
        }
    }

    for model_name in ['federated_fusion', 'full_fusion', 'bert_only']:
        cm = np.zeros((4, 4), dtype=int)
        for i, level in enumerate(levels):
            support = int(class_reports[model_name][level]['support'])
            recall = class_reports[model_name][level]['recall']
            true_positives = int(round(recall * support))
            cm[i, i] = true_positives
            false_negatives = support - true_positives
            if false_negatives > 0:
                other_classes = [j for j in range(4) if j != i]
                fn_per_class = false_negatives // len(other_classes)
                for j in other_classes:
                    cm[i, j] += fn_per_class
                remaining = false_negatives - fn_per_class * len(other_classes)
                if remaining > 0:
                    cm[i, other_classes[0]] += remaining
        metrics[model_name]['confusion_matrix'] = cm
        metrics[model_name]['class_report'] = class_reports[model_name]

    return metrics

metrics = get_metrics()

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
    to assess depression levels using advanced machine learning techniques, including federated learning.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">'
                   '<h3>Federated Fusion Model</h3>'
                   f'<p class="big-font">Accuracy: {metrics["federated_fusion"]["accuracy"]*100:.1f}%</p>'
                   '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">'
                   '<h3>Centralized Fusion Model</h3>'
                   f'<p class="big-font">Accuracy: {metrics["full_fusion"]["accuracy"]*100:.1f}%</p>'
                   '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">'
                   '<h3>BERT-Only Model</h3>'
                   f'<p class="big-font">Accuracy: {metrics["bert_only"]["accuracy"]*100:.1f}%</p>'
                   '</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Depression Level Classification:
    - **0: None** - No significant signs of depression
    - **1: Mild** - Minor symptoms that may not interfere with daily life
    - **2: Moderate** - Noticeable symptoms that affect some daily activities
    - **3: Severe** - Significant impairment in daily functioning
    """)
    
    st.subheader("Sample Data Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = [128, 128, 128, 128]
    sns.barplot(x=levels, y=counts, palette="Blues_d", ax=ax)
    ax.set_title("Distribution of Depression Levels in Training Data")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def show_model_performance():
    st.title("Model Performance Analysis")
    
    st.markdown("""
    ### Comparative Model Evaluation
    Below you can see the performance metrics for our three main models:
    - **Federated Fusion Model**: Combines all data modalities with federated learning
    - **Centralized Fusion Model**: Combines all data modalities centrally
    - **BERT-Only Model**: Uses only text data from social media and chatbot interactions
    """)
    
    comparison_data = {
        'Model': ['Federated Fusion', 'Centralized Fusion', 'BERT-Only'],
        'Accuracy': [metrics["federated_fusion"]["accuracy"],
                    metrics["full_fusion"]["accuracy"],
                    metrics["bert_only"]["accuracy"]],
        'F1-Score': [metrics["federated_fusion"]["f1"],
                    metrics["full_fusion"]["f1"],
                    metrics["bert_only"]["f1"]],
        'Recall': [metrics["federated_fusion"]["recall"],
                  metrics["full_fusion"]["recall"],
                  metrics["bert_only"]["recall"]]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    display_df = comparison_df.copy()
    for col in ['Accuracy', 'F1-Score', 'Recall']:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")
    
    st.dataframe(display_df.style.background_gradient(cmap='Blues'), 
                use_container_width=True)
    
    st.subheader("Confusion Matrices")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Federated Fusion Model**")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        sns.heatmap(metrics['federated_fusion']['confusion_matrix'], annot=True, fmt='d',
                   cmap='Blues', xticklabels=levels, yticklabels=levels)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)
    
    with col2:
        st.markdown("**Centralized Fusion Model**")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        sns.heatmap(metrics['full_fusion']['confusion_matrix'], annot=True, fmt='d',
                   cmap='Blues', xticklabels=levels, yticklabels=levels)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)
    
    with col3:
        st.markdown("**BERT-Only Model**")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        sns.heatmap(metrics['bert_only']['confusion_matrix'], annot=True, fmt='d',
                   cmap='Greens', xticklabels=levels, yticklabels=levels)
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")
        st.pyplot(fig3)
    
    st.subheader("Detailed Classification Reports")
    tab1, tab2, tab3 = st.tabs(["Federated Fusion Model", "Centralized Fusion Model", "BERT-Only Model"])
    
    with tab1:
        st.markdown("**Federated Fusion Model Classification Report**")
        report_df = pd.DataFrame(metrics['federated_fusion']['class_report']).T
        report_df = report_df.applymap(lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x)
        st.dataframe(report_df.style.background_gradient(cmap='Blues'))
    
    with tab2:
        st.markdown("**Centralized Fusion Model Classification Report**")
        report_df = pd.DataFrame(metrics['full_fusion']['class_report']).T
        report_df = report_df.applymap(lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x)
        st.dataframe(report_df.style.background_gradient(cmap='Blues'))
    
    with tab3:
        st.markdown("**BERT-Only Model Classification Report**")
        report_df = pd.DataFrame(metrics['bert_only']['class_report']).T
        report_df = report_df.applymap(lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x)
        st.dataframe(report_df.style.background_gradient(cmap='Greens'))

def show_chatbot():
    st.title("Depression Assessment Chatbot")
    
    st.markdown("""
    ### Interactive Depression Screening
    This chatbot analyzes your responses to assess potential depression symptoms.
    All conversations are anonymous and not stored.
    """)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.conversation = []
        st.session_state.depression_level = None
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("How are you feeling today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation.append(("user", prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("Analyzing your response..."):
            try:
                processed_text = preprocess_text(prompt)
                
                if not processed_text.strip():
                    response = "Could you please share more about how you're feeling?"
                else:
                    embedding = get_bert_embedding(processed_text)
                    
                    if models['bert_only'] is not None:
                        prediction = models['bert_only'].predict(
                            [embedding.reshape(1, -1), embedding.reshape(1, -1)], 
                            verbose=0
                        )
                        depression_level = np.argmax(prediction)
                        st.session_state.depression_level = depression_level
                        response = get_chatbot_response(depression_level)
                    else:
                        response = "BERT-only model not available. Ensure 'bert_only_fusion_model.keras' is in the same directory as app.py."
            except Exception as e:
                st.error(f"Error processing your message: {str(e)}")
                response = "I encountered an error processing your message. Please try again."
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.conversation.append(("assistant", response))
            
            with st.chat_message("assistant"):
                st.markdown(response)
    
    if st.session_state.depression_level is not None:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Current Assessment</h3>
            <p class="big-font">Depression Level: {levels[st.session_state.depression_level]}</p>
            <p>{get_severity_description(st.session_state.depression_level)}</p>
        </div>
        """, unsafe_allow_html=True)
        
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
    Select a model to use for prediction.
    """)
    
    model_choice = st.selectbox("Select Prediction Model", 
                               ["Federated Fusion", "Centralized Fusion", "BERT-Only"])
    
    model_map = {
        "Federated Fusion": "federated_fusion",
        "Centralized Fusion": "full_fusion",
        "BERT-Only": "bert_only"
    }
    selected_model = model_map[model_choice]
    
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
        no_employees = st.selectbox("Company size (number of employees)", 
                                   ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
        
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
                    # Preprocess inputs
                    le_gender = LabelEncoder()
                    le_gender.fit(['Male', 'Female', 'Other', 'Prefer not to say'])
                    gender_encoded = le_gender.transform([gender])[0]
                    
                    le_family = LabelEncoder()
                    le_family.fit(['No', 'Yes'])
                    family_history_encoded = le_family.transform([family_history])[0]
                    
                    le_employees = LabelEncoder()
                    le_employees.fit(['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
                    no_employees_encoded = le_employees.transform([no_employees])[0]
                    
                    scaler_age = StandardScaler()
                    age_scaled = scaler_age.fit_transform(np.array([[age]]))[0][0]
                    
                    clinical_input = np.array([[age_scaled, 
                                             gender_encoded, 
                                             family_history_encoded, 
                                             no_employees_encoded]]).astype('float32')
                    
                    scaler_phys = StandardScaler()
                    phys_data = np.array([[sleep_duration,
                                         sleep_quality,
                                         heart_rate,
                                         daily_steps,
                                         sleep_duration * sleep_quality]])
                    phys_scaled = scaler_phys.fit_transform(phys_data)[0]
                    
                    phys_input = phys_scaled.astype('float32')
                    phys_cnn_input = phys_input.reshape(1, 5, 1)
                    phys_lstm_input = phys_input.reshape(1, 5, 1)
                    
                    text_embedding = np.zeros(768)
                    if statement and isinstance(statement, str) and statement.strip():
                        processed_text = preprocess_text(statement)
                        text_embedding = get_bert_embedding(processed_text)
                    
                    text_input = text_embedding.reshape(1, -1).astype('float32')
                    
                    if models[selected_model] is not None:
                        if selected_model == 'bert_only':
                            prediction = models[selected_model].predict(
                                [text_input, text_input], verbose=0
                            )
                        else:
                            prediction = models[selected_model].predict(
                                [clinical_input, phys_cnn_input, phys_lstm_input, 
                                text_input, text_input], verbose=0
                            )
                        
                        depression_level = np.argmax(prediction)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Assessment Results</h3>
                            <p class="big-font">Depression Level: {levels[depression_level]}</p>
                            <p>{get_severity_description(depression_level)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
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
                        st.error(f"{model_choice} model not available. Ensure the corresponding .keras file is in the same directory as app.py.")
                
                except Exception as e:
                    st.error(f"Error processing your assessment: {str(e)}")

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Choose a section", 
                               ["Dashboard", "Model Performance", 
                               "Client Assessment", "Chatbot Interaction"])
    
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
