import streamlit as st
from transformers import pipeline
from PIL import Image

def sentence_classifier(sentence):
    classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
    sequence_to_classify = sentence
    candidate_labels = ['camera','batterry','performance','display']
    return classifier(sequence_to_classify, candidate_labels)

def sentiment_analyser(sentence):
    sentiment = pipeline("sentiment-analysis")
    return sentiment(sentence)

st.title('E-Commerce Reviewer')
st.image('https://imgeng.jagran.com/images/2023/mar/flipkart-big-bachat-dhamaal-sale1677826995237.jpg')
sentence = st.text_input('Enter the sentence')

if st.button('Submit Review'):
    
    CatSent = sentence_classifier(sentence)
    st.write(CatSent)
    predcat = CatSent['labels'][0]
    probcat = CatSent['scores'][0]
    st.write('Predicted Category: ',predcat)
    st.write('Confidence: ',probcat)
    
    ClassSent = sentiment_analyser(sentence)
    st.write(ClassSent)
    predlabel = ClassSent[0]['label']
    problabel = ClassSent[0]['score']
    st.write('Predicted Sentiment: ',predlabel)
    st.write('Confidence: ',problabel)
    
    if predlabel == 'POSITIVE':
        
        st.success(f"""Sentence : {sentence} \n
                   Predicted Category : {predcat} \n
                   Predicted Sentiment : {predlabel}""")

        st.balloons()
        
    elif predlabel == 'NEGATIVE':
        
        st.title("Negative Review 🔥🔥🔥🔥🔥" )
        
        st.error(f"""Sentence : {sentence} \n
                   Predicted Category : {predcat} \n
                   Predicted Sentiment : {predlabel}""")

        
    