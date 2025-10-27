import torch
import streamlit as st
import zipfile
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

base_model_dir = 'saved_model'  

def find_model_directory(base_path):
    if any(f in os.listdir(base_path) for f in ['pytorch_model.bin', 'model.safetensors']):
        return base_path
    
    for root, dirs, files in os.walk(base_path):
        if 'pytorch_model.bin' in files or 'model.safetensors' in files:
            return root
        if root.count(os.sep) - base_path.count(os.sep) > 2:
            dirs[:] = []
    return None

extraction_path = base_model_dir
try:
    if not os.path.isdir(extraction_path):
        with zipfile.ZipFile('saved_model.zip', 'r') as zip_ref:
            zip_ref.extractall(path=extraction_path)
    else:
        st.info('Model Exists. Skipping extraction.')
except FileNotFoundError:
    st.error('File not found!')

final_model_dir = find_model_directory(extraction_path)

@st.cache_resource
def load_model_tokenizer(model_dir):
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model_tokenizer(final_model_dir)

def prefix_text(input_text):
    return 'paraphrase: ' + input_text

def generate_paraphrase(input_text, model, tokenizer, device, max_length=256, num_beams=4, num_return_sequences=4, top_k=100, top_p=1.0, temperature=2.5):
    input_text = prefix_text(input_text)
    inputs = tokenizer(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_length + 20,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        early_stopping=True,
        do_sample=True
    )

    paraphrase_text = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
    return paraphrase_text

# Streamlit UI
# st.set_page_config(page_title='Paraphrase Generator', layout='centered')
st.title('📝 T5 Paraphrase Generator')

text_input = st.text_area('Enter text to paraphrase:', height=80)

if st.button('Generate'):
    if text_input.strip():
        paraphrases = generate_paraphrase(text_input, model, tokenizer, device)
        st.subheader("🔁 Generated Paraphrases:")
        for i, p in enumerate(paraphrases, 1):
            st.write(f"**{i}.** {p}")
    else:
        st.warning("⚠️ Please enter some text first.")