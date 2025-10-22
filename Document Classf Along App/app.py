import numpy as np
import pickle
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pypdf
import fitz # For rendering PDFs as images.
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from PIL import Image


# --- Model and NLTK Setup (Keep as is) ---

# NOTE: In a real application, you'd load the model like this:
# model = AutoModelForSequenceClassification.from_pretrained('path/to/your/model')
# For demonstration, we assume loading from scratch for clarity, but you'll need the trained weights.
try:
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    model_name = 'distilbert-base-uncased'
    token = AutoTokenizer.from_pretrained(model_name)
    # Assuming your fine-tuned model weights are saved/loaded correctly here
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
except FileNotFoundError:
    st.error("Model or label encoder files not found. Please ensure 'label_encoder.pkl' and the fine-tuned model weights are in the correct directory.")
    st.stop()


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmantizer = WordNetLemmatizer()

def text_preprocessing(text):
    
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\r\w', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(lemmantizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text


def predict(text):
    text = text_preprocessing(text)
    inputs = token(text, return_tensors='pt', truncation=True, padding=True, max_length= 128)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = np.argmax(outputs.logits, axis=1)
    return label_encoder.inverse_transform(predictions)[0]

# --- FIX: Update helper functions to accept raw bytes ---

def extract_text_from_pdf(file_bytes):
    """Extracts text from PDF bytes."""
    # fitz.open can take the raw bytes directly
    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ''
    for page in range(len(pdf_doc)):
        # Using page.get_text() is often cleaner
        page_text = pdf_doc[page].get_text() 
        text += page_text + '\n'
    
    pdf_doc.close()
    return text


def render_pdf_as_images(file_bytes):
    """Renders PDF bytes as a list of PIL Images."""
    # fitz.open can take the raw bytes directly
    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        # set zoom/scale for better quality (optional)
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) 
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    pdf_doc.close()
    return images

# --- Streamlit App ---

st.set_page_config(page_title= 'Document Classifier', layout= 'centered')
st.title('Document Classifier')
st.write('Upload a document (PDF or TXT) to classify its content.')

uploaded_file = st.file_uploader('Choose a file', type=['pdf', 'txt'], 
                                 help='Limit 200MB per file • PDF, TXT')

if uploaded_file is not None:
    # --- CRITICAL FIX: Read file content ONCE ---
    # Use getvalue() to read the raw bytes/content without moving the file pointer of the uploaded_file object
    file_bytes = uploaded_file.getvalue() 
    file_type = uploaded_file.type

    text = ""
    
    if file_type == 'application/pdf':
        text = extract_text_from_pdf(file_bytes)
    elif file_type == 'text/plain':
        # Decode the bytes to string for TXT files
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = file_bytes.decode('latin-1') # Fallback if UTF-8 fails

    
    if text: # Ensure text was successfully extracted
        prediction = predict(text)

        st.subheader('Prediction Result')
        # Display the result
        st.info(f'The document is classified as: **{prediction}**')
        
        # --- Document Preview Section ---
        st.subheader('Document Preview')

        if file_type == 'application/pdf':
            # Use the same file_bytes for rendering
            pdf_images = render_pdf_as_images(file_bytes) 
            for img in pdf_images:
                st.image(img, use_container_width=True)
        elif file_type == 'text/plain':
            # For TXT files, show the content directly
            st.code(text, language='text')

    else:
        st.error("Could not extract text from the uploaded file.")