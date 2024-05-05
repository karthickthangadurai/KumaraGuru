from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import streamlit as st

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# prepare image + question
url = st.text_input("Enter the url of the image")

try:
    image = Image.open(requests.get(url, stream=True).raw)
    st.image(image)
except:
    pass

text = st.text_input("Enter the question")

if st.button("Submit") and text is not None and image is not None:

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    st.title(f"Predicted answer: {model.config.id2label[idx]}")
    