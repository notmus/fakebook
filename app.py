# this is needed so it can write to app.py as streamlit runs on app.py

import streamlit as st
import pickle
from pycaret.regression import load_model
from sklearn.preprocessing import LabelEncoder, TargetEncoder
import numpy as np
import pandas as pd
from streamlit_extras.let_it_rain import rain

# layout of the page
def main():
    # Set page title and favicon
    st.set_page_config(page_title="Fakebook", page_icon="")

    # Page title
    st.title("Welcome to Fakebook")

    # Add some text
    st.write("This is a Facebook Marketplace Scam Predictor.")

    # Add a sidebar
    st.sidebar.title("Instructions")
    st.sidebar.write("Key in the Product Name, Price, Condition, and User Joined Year. Once those have been inputted, press Predict.")

    # Product Name input
    Name = st.text_input("Product Name:", "")

    # Price input
    Price = st.text_input("Price:", "")

    # Condition
    Condition = st.selectbox("Select Condition", ["New", "Used-like-new"])
    st.write(f"Condition selected: {Condition}")

    # Year
    Joined_Date = st.selectbox("Select User Joined Year ", ["2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"])
    st.write(f"Year selected: {Joined_Date}")

    # Loading the model
    if st.button("Predict"):
        # Load the saved model
        model = load_model('/content/drive/MyDrive/Colab Notebooks/ML/model')  # Replace 'model' with the path to your saved model

        # Create target encoder (assuming your target variable is 'label')
        encoder = TargetEncoder()

        # Load csv file that maps original product name to encoded product name
        transformed_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/ML/product_name_mapping.csv")

        # Get the transformed data from PyCaret (assuming it's stored in 'transformed_data')
        #transformed_data = model.data.copy()  # Make a copy to avoid modifying original data

        # Fit the encoder on the entire transformed data (including target)
        encoder.fit(transformed_data.iloc[:,0:1], transformed_data.iloc[:,1:2])

        # Transform the product name (assuming it's the first column)
        encoded_name = encoder.transform(np.array([[Name]]))[0]

        # Convert Price to float
        Price = float(Price)

        # Convert Joined Date to float
        Joined_Date = float(Joined_Date)

        # Instantiate the encoder
        encoder = LabelEncoder()

        # Fit the encoder to the "Condition" column
        encoder.fit(["New", "Used-like-new"])

        # Transform the "Condition" column using the fitted encoder
        encoded_condition = encoder.transform([Condition])[0]

        # Prepare input data for prediction
        input_df = pd.DataFrame({'Name': [Name], 'Price': [Price], 'Condition': encoded_condition, 'Joined Date': Joined_Date})

        # Make prediction
        prediction = model.predict(input_df)

        #st.write(prediction)

        # Determine prediction label
        if prediction[0] == "scam":
            #rain(emoji="☠️", font_size=54, falling_speed=1,animation_length="infinite")
            prediction_label = "Scam"
        else:
            prediction_label = "Legitimate"

        # Display the prediction
        st.subheader('Prediction')
        st.write('The listing is:', prediction_label)

if __name__ == "__main__":
    main()
