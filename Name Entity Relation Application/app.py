import torch
import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import zipfile
import os

# --- Define CUSTOM LABEL MAPPING (User provided fix) ---
# Assuming the labels are swapped: the model seems to be using LABEL_4 for Persons (wrongly) and LABEL_3 for ORG (wrongly)
CUSTOM_LABEL_MAP = {
    'LABEL_4': 'ORG',  # SpaceX, Tesla
    'LABEL_3': 'PER',  # Elon Musk
    'O': 'O'
}   

# --- Words to force to 'O' (Outside) entity type ---
OVERRIDE_TO_O_WORDS = {'leads', 'and', 'the', 'a', 'is', 'are', 'was', 'were', '.'}


# --- Determine the target device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Unzip the model if not already extracted ---
if not os.path.exists("ner_finetuned_model"):
    try:
        with zipfile.ZipFile("ner_finetuned_model.zip", "r") as zip_ref:
            zip_ref.extractall("ner_finetuned_model")
    except FileNotFoundError:
        st.stop()


# --- Load model and tokenizer ---
try:
    model = AutoModelForTokenClassification.from_pretrained(
        "ner_finetuned_model", 
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained("ner_finetuned_model")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# --- Label mapping ---
id_to_label = model.config.id2label

# --- Function to perform NER ---
def ner_detection(sentences):
    model.eval()

    results = []
    for text in sentences:
        tokens = tokenizer(text, return_tensors='pt', truncation=True, is_split_into_words=False)
        tokens = {k: v.to(device) for k, v in tokens.items()} # Move to device e.g(GPU/CPU)


        # Doing No Gradient Calculation.
        with torch.no_grad():
            output = model(**tokens)

        predictions = np.argmax(output.logits.detach().cpu().numpy(), axis=2)
        token_list = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
        labels = [id_to_label[label] for label in predictions[0]]

        sentence_result = []
        for token, label in zip(token_list, labels):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                sentence_result.append((token, label))
        results.append(sentence_result)

    return results

# --- Streamlit UI ---
st.set_page_config(page_title="NER App", page_icon="🤖", layout="centered")
st.title("Named Entity Recognition (NER) Application")
user_input = st.text_area("Enter your text here:", 
                          value="Elon Musk leads SpaceX and Tesla.", 
                          height=100)

if st.button("Detect Entities"):
    raw_sentences = user_input.split('\n')
    sentences = []
    for line in raw_sentences:
        sentences.extend([s.strip() for s in line.split('.') if s.strip()])
    if not sentences and user_input.strip():
        sentences = [user_input.strip()]
        
    if sentences:
        detect_entities_list = ner_detection(sentences)
        
        final_entities = []
        for sentence_idx, entities in enumerate(detect_entities_list):
            current_entity_type = "O"
            current_entity_tokens = []
            
            for token, label_with_prefix in entities:
                
                # 1. Separate prefix (B-, I-) and clean label
                parts = label_with_prefix.split('-')
                prefix = parts[0]
                clean_label_raw = parts[-1] 
                
                # 2. Use the custom map to replace generic labels with meaningful names
                clean_label = CUSTOM_LABEL_MAP.get(clean_label_raw, clean_label_raw)
                
                # 3. Process the token for joining
                # Keep the token as is for now, without removing '##' or stripping spaces
                word_part = token 
                
                # Aggressively apply override to tokens that are definitely non-entities, 
                # even if the model mislabeled them. We use the cleaned version for comparison.
                if word_part.strip().replace('##','').lower() in OVERRIDE_TO_O_WORDS:
                    clean_label = "O"
                
                # --- IOB-AWARE Entity Reassembly ---

                # Condition for starting a NEW entity (B- or a non-O label after a different type)
                is_new_entity_start = clean_label != "O" and (prefix == 'B' or clean_label != current_entity_type)

                if is_new_entity_start:
                    
                    # 1. Save the previous entity if one exists
                    if current_entity_type != "O":
                        # The join and strip handles concatenation and final cleaning
                        final_entities.append((
                            f"Sentence {sentence_idx + 1}",
                            "".join(current_entity_tokens).strip().replace('##', ''),
                            current_entity_type
                        ))
                    
                    # 2. Start the new entity
                    current_entity_type = clean_label
                    # Start entity with the token
                    current_entity_tokens = [word_part] 
                
                # CASE: Continuation of the current entity (I- label with the same type)
                elif clean_label != "O" and clean_label == current_entity_type:
                    
                    # Fix: Inject a space before the token if it's not a subword continuation (##) 
                    # AND it doesn't already contain a leading space (for BPE/SentencePiece tokenizers).
                    # This handles the ElonMusk concatenation.
                    is_subword_or_spaced = word_part.startswith('##') or word_part.startswith(' ')
                    
                    if not is_subword_or_spaced and current_entity_tokens:
                          # Inject a space only if a subword or leading space marker is absent.
                          current_entity_tokens.append(' ')
                          
                    # Append the current token part (which is still raw, to be cleaned at join time)
                    current_entity_tokens.append(word_part)

                # CASE: Entity Ends ('O')
                elif clean_label == "O":
                    # 1. Save the previous entity if one was being tracked
                    if current_entity_type != "O":
                        final_entities.append((
                            f"Sentence {sentence_idx + 1}",
                            # FINAL CLEANING STEP: Join tokens and remove '##' subword markers
                            "".join(current_entity_tokens).strip().replace('##', ''),
                            current_entity_type
                        ))
                    
                    # 2. Reset tracking.
                    current_entity_type = "O"
                    current_entity_tokens = []

            # After loop, save the last entity if one exists
            if current_entity_type != "O":
                final_entities.append((
                    f"Sentence {sentence_idx + 1}",
                    "".join(current_entity_tokens).strip().replace('##', ''),
                    current_entity_type
                ))
    
        
        # Display all entities except 'O' tokens and empty strings.
        df_data = [row for row in final_entities if row[2] != 'O' and row[1]]

        df = pd.DataFrame(df_data, columns=["Sentence", "Entity Text", "Entity Type"])
        
        st.write("### Detected Entities:")
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("No named entities were detected (or all were filtered as 'O').")

    else:
        st.warning("Please enter some text to analyze.")