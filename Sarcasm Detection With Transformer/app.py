import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

TF_ENABLE_ONEDNN_OPTS = 1

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = TFAutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', num_labels=2, from_pt=True)
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title= '🧐Sarcasm Detection App', page_icon= '🤖', layout= 'centered')
st.title('🧐Sarcasm Detection App')
st.write('Write any sentence & i will tell you if it is sarcastic or not!')

user_input = st.text_area('Enter your text here:', height=150)
if st.button('Predict'):
    if user_input.strip() == '':
        st.write('Please enter some text to analyze.')
    else:

        input_sentence = tokenizer(user_input, truncation= True, padding= True, return_tensors= 'tf')
        pred = model(input_sentence)[0]
        prob = tf.nn.softmax(pred, axis= -1).numpy()
        ids = tf.argmax(prob, axis= 1)

        if ids[0] == 0:
            st.write('The sentence is not sarcastic. 🙂')
        else:
            st.write('The sentence is sarcastic. 😐')