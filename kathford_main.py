import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

#load the model
model = joblib.load(r"C:\Python\model_kathford.pt")

#process image
def image_prcoess(img):
    img_pil = Image.open(img)
    img_ary = np.array(img_pil)
    img_flat = img_ary.flatten()
    df_t = pd.DataFrame(img_flat).T
    #make predictions
    p=model.predict(df_t)
    return p

#Streamlit code
st.title("Shoes Classification, Machine Learning")
file=st.file_uploader("Upload your file" ,type=['jpg', 'jpeg'])
try:
    if file is not None:
        i = Image.open(file)
        st.write(i)
        pr=image_prcoess(file)
        st.write(f"The predicted brand is {pr}")
    else:
        st.write("Empty file")
except Exception as e:
    st.write("{e}")