import streamlit as st
from transformers import VitsModel, AutoTokenizer
import torch
from IPython.display import Audio

st.image(r"D:\VS_Code\KumaraGuru\T2S.jpeg")

st.title("Text to Speech")
text = st.text_input("Enter The Text")

if st.button("Submit") and text is not None:
    
    model = VitsModel.from_pretrained("facebook/mms-tts-tam")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tam")


    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    output = Audio(output.numpy(), rate=model.config.sampling_rate)
    
    with open(r'D:\\VS_Code\\KumaraGuru\\test.wav', 'wb') as f:
        f.write(output.data)
        
    st.audio(r'D:\\VS_Code\\KumaraGuru\\test.wav')
    
