import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import matplotlib.pyplot as plt
import contractions
import re
from wordcloud import WordCloud
import string

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}?\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    #st.write(chain)
    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    # st.write(response)
    return response["output_text"]
    # print(response)
    # st.write("Reply: ", response["output_text"])

#===============================For word Cloud=============================================#
#===========================================================================================#
def to_lowercase(text):
    return text.lower()


#to remove the contractions shorten the words
def expand_contractions(text):
    expanded_words = [] 
    for word in text.split():
       expanded_words.append(contractions.fix(word)) 
    return ' '.join(expanded_words)
# print(expand_contractions("Don't is same as do not"))
# >> Do not is same as do not

#Removing Mentions and Hashtags
def remove_mentions_and_tags(text):
    text = re.sub(r'@\S*', '', text)
    return re.sub(r'#\S*', '', text)
# testing the function on a single sample for explaination
# print(remove_mentions_and_tags('Some random @abc and#def'))
# >> Some random and 

#Removing Special Characters
def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)
# print(remove_special_characters(“007 Not sure@ if this % was #fun! 558923 What do# you think** of it.? $500USD!”))
# >> '007 Not sure if this  was fun! 558923 What do you think of it.? 500USD!'

#Removing Digits
def remove_numbers(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    
    return re.sub(pattern, '', text)
# print(remove_numbers(“You owe me 1000 Dollars”))
# >> You owe me Dollars

# Removing Puncuations

def remove_punctuation(text):
    return ''.join([c for c in text if c not in string.punctuation])
# remove_punctuation('On 24th April, 2005, "Me at the zoo" became the first video ever uploaded to YouTube.')
# >> On 24th April 2005 Me at zoo became the first video ever uploaded to Youtube


def generate_word_cloud(pdf_docs):
    
    if pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text = to_lowercase(raw_text)
        text = expand_contractions(text)
        text = remove_mentions_and_tags(text)
        text = remove_special_characters(text)
        #text = remove_numbers(text)
        text = remove_punctuation(text)
        
        # Create and generate a word cloud image:
        wordcloud = WordCloud().generate(text)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig = plt.show()
        st.pyplot(fig)
        
    else:
        st.write("Upload your CricetPDF Files To See World Cloud")
        

def main():
    
    st.set_page_config("Chat PDF")
    
    st.markdown("<h1 style='text-align: center; color: green;'>🏏GUVICRIC Assistant💁</h1>", unsafe_allow_html=True)
    #st.header("🏏GUVICRIC Assistant💁")
    with st.sidebar:
        
        st.markdown("<h1 style='text-align: center; color: green;'>Main Manu</h1>", unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.toast('Your documents and vector store was saved!', icon='😍')
                st.success("Done")
    
    with st.spinner("Generating Word Cloud..."):
        generate_word_cloud(pdf_docs)
    
    user_question = st.text_input("Ask a Question To Assitant")

    if user_question is not None and st.button("Submit"):

        response=user_input(user_question)
        # st.markdown("Reply: ", response)
        st.markdown(f"<h4 style='text-align: center; color: green;'>Results: {response}</h4>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
