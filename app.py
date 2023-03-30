import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the h5 model file
model = load_model('toxicity.h5')

def main():
    st.title('Toxicity Detector')

        # Create a text input for the line
    line = st.text_input("Enter a line here:")

    # Load the tokenizer
    tokenizer = Tokenizer(num_words=10000)

    # Create a button to check the toxicity of the line
    if st.button('Check Toxicity'):
        # Vectorize the input line
        input_data = tokenizer.texts_to_sequences([line])
        input_data = pad_sequences(input_data,maxlen=1000)

        # Predict the toxicity of the line using the model
        results = model.predict(input_data)

        # Create a DataFrame to display the toxicity results
        cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        df = pd.DataFrame(results, columns=cols)

        # Format the results as a string
        text = ''
        for idx, col in enumerate(df.columns):
                text += '{}: {}\n'.format(col, df[col][0]>0.5)

        # Display the results as text
        st.write(results)

# Load the tokenizer

# # Define the Streamlit app
# def app():
#     st.set_page_config(page_title="Toxicity Checker")

#     # Set the app header
#     st.write("""
#     # Toxicity Checker
#     Please enter a line to check its toxicity.
#     """)

    # Create a text input for the line
    # line = st.text_input("Enter a line here:")

    # # Create a button to check the toxicity of the line
    # if st.button('Check Toxicity'):
    #     # Vectorize the input line
    #     input_data = tokenizer.texts_to_sequences([line])
    #     input_data = pad_sequences(input_data, maxlen=100)

    #     # Predict the toxicity of the line using the model
    #     results = model.predict(input_data)

    #     # Create a DataFrame to display the toxicity results
    #     cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    #     df = pd.DataFrame(results, columns=cols)

    #     # Format the results as a string
    #     text = ''
    #     for idx, col in enumerate(df.columns):
    #         text += '{}: {}\n'.format(col, df[col][0]>0.5)

    #     # Display the results as text
    #     st.write(text)
if __name__ == '__main__':
    main()