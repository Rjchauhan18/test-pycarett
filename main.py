import streamlit as st
from pycaret.classification import *
import pandas as pd 

# Title of the application
st.title('PyCaret Streamlit Example')

# Load dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    # Display the dataset
    st.write(data)

    # Build and display model
    st.subheader('Model Building')

    target = st.selectbox("Select target variable", data.columns)

    if st.button('Build Model'):
        # Progress bar
        progress_bar = st.progress(0)

        with st.spinner('Building model...'):
            # Build model
            exp = setup(data, target=target)
            best_model = compare_models()

        # Update progress bar
        progress_bar.progress(100)

        st.write(best_model)

    # Make predictions
    st.subheader('Make Predictions')

    if 'best_model' in locals():
        prediction_data = data.drop(target, axis=1).head(5)
        predictions = predict_model(best_model, data=prediction_data)
        st.write(predictions)
