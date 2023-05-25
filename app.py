import streamlit as st
import pandas as pd

from tensorflow import keras
from  prediction import get_prediction

model = keras.models.load_model("models")

st.set_page_config(page_title="Higgs Boson Event Detection App", page_icon="🌠", layout="wide")

st.markdown("<h1 style='text-align: center;'>Higgs Boson Event Detection App 🌠</h1>", unsafe_allow_html=True)
st.markdown("This project is part of my personal portfolio.")
st.markdown("In this, an attempt is made to classify whether the given event was a signal or a background noise in the process of decay for Higgs particle acceleration.")
st.markdown("The target feature is **:red[Label]** which is a binary variable. The task is to classify this variable based on the other 31 features.")
st.markdown("The metric used for evaluation is **:green[Precision]**")
st.write("You can check the source code on [GitHub](https://github.com/LuisAnibalVasquez/Higgs-Boson-Event-Detection)")

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("You can use this template file to make prediction")
    with col2:
        with open("Templete.csv", "rb") as file:
            btn = st.download_button(label="Download",
                                    data=file,
                                    file_name="Templete.csv",
                                    mime="text/csv"
                                    )



def main():
    with st.form('prediction_form'):
        
        st.subheader("Enter the input file for prediction:")

        uploaded_file = st.file_uploader("Choose a file")

        if uploaded_file is not None:

            dataframe = pd.read_csv(uploaded_file)
            dataframe.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

        submit = st.form_submit_button("Predict")

        if submit:
            data = dataframe.to_numpy().reshape(1,-1)
            pred = get_prediction(data=data, model=model)

            a =  round(float(pred[0]) )

            st.subheader(f"The predicted event is:  {a:20f}")            
            st.markdown("A probability of zero (0) indicates that the event is a  **:green[Signal]**.")
            st.markdown("A probability of one (1) indicates that the event is a  **:red[Background]**.")            

if __name__ == '__main__':
    main()