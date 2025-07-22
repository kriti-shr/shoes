import streamlit as st 
import joblib 
import pandas as pd

# laod the model 
model = joblib.load(r'C:\Python\model_heart.pt')

# function to encode and predcit data 
def predict(data): 
    data=data.dropna()
    df=data.copy()
    df = encode_ecg(df)
    df = encode_ang(df)
    df = encode_sex(df)
    df = encode_chestPain(df)
    df = encode_slope(df)
    
    # df = decode_heart(data, df)
    # after encode 
    df['predicted']=model.predict(df)

    return df

def encode_ecg(df):
    enc_ecg = {'Normal':1, 'ST':2, 'LVH':3}
    df=df.drop('HeartDisease',axis=1)
    df['RestingECG']=df['RestingECG'].map(enc_ecg)
    return df

def encode_ang(df):
    enc_ang = {'N':0, 'Y':1}
    df['ExerciseAngina']=df['ExerciseAngina'].map(enc_ang)
    return df

def encode_sex(df):
    enc_sex = {'M':1, 'F':0}
    df['Sex']=df['Sex'].map(enc_sex)
    return df

def encode_chestPain(df):
    enc_chest = {'ATA':1, 'NAP':2, 'ASY':3, 'TA':4}
    df['ChestPainType']=df['ChestPainType'].map(enc_chest)
    return df

def encode_slope(df):
    enc_slope = {'Up':1, 'Flat':2, 'Down':3}
    df['ST_Slope']=df['ST_Slope'].map(enc_slope)
    return df

def decode_heart(df):
    enc_heart = {0:'No Heart Disease', 1:'Heart Disease'}
    df['predicted']=df['predicted'].map(enc_heart)
    return df

# title of applciation 
st.title("Heart Disease Prediction")
file=st.file_uploader("Upload Your file", type='csv')
try:
    if file is not None:
        data=pd.read_csv(file)
        st.write("first five rows")
        st.write(data.head())
        df_p=predict(data)
        df_p = decode_heart(df_p)

        st.write("Data with prediction")
        st.write(df_p)
    else:
        st.write("Empty file cannot be read")
except Exception as e: 
    st.write(f"Error {e} occured")

finally: 
    st.write("Thank you for using our service ")