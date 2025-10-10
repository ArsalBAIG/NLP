import streamlit as st
import pickle
import tensorflow

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(page_title= 'Quora Insincere Questions App', page_icon= '🤖', layout= 'centered')

model = load_model('model.h5')

max_features = 10000
max_len = 100

token = Tokenizer(num_words= max_features)

st.title("🧠 InSincere Question Quora App.")
st.write("Detect whether a Quora question is **sincere** or **insincere** using an LSTM model 💬")

user_input = st.text_area('Enter you question here: ', height= 100)

if st.button('Predict!'):
    if user_input.strip() == '':
        st.write('Please enter some text to analyze.')
    else:
        token.fit_on_texts([user_input])
        seq = token.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen= max_len)

        pred = model.predict(padded)
        if pred[0] > 0.5:
            st.write('The question is sincere. 😐')
        else:
            st.write('The question is InSincere. 🙂')
