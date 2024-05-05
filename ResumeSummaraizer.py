import streamlit as st
import base64
import PyPDF2
from transformers import pipeline
import torch
summarizer = pipeline("summarization","pszemraj/long-t5-tglobal-base-16384-book-summary",
    device=0 if torch.cuda.is_available() else -1,)

st.title("Resume Summarizer Application")

#save file to temp folder

def clear_submit():
    st.session_state["submit"] = False
def displayPDF(uploaded_file):
    
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

    # Embed PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)
    
uploaded_file = st.file_uploader(
    "Upload file", type=["pdf"], 
    help="Only PDF files are supported", 
    on_change=clear_submit)
#save file as temp
if uploaded_file is not None:
    with open("resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())


if uploaded_file is not None:
    if st.button("Summarize"):
        displayPDF(uploaded_file)

        pdfFileObj = open("resume.pdf", 'rb')
        
        # creating a pdf reader object
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        
        # printing number of pages in pdf file
        text=""
        for i in range(len(pdfReader.pages)):
            text=text+pdfReader.pages[i].extract_text()
            st.write()
        st.title("Summarized Resume")
        st.write(summarizer(text,max_length=500, min_length=150)[0]['summary_text'])
        
        # creating a page object
        
        
        # extracting text from page
        
        
        # closing the pdf file object
        pdfFileObj.close()







