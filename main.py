import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

model = pickle.load(open('model.pkl', 'rb'))

st.title("Amazon Product Review Sentiment Analysis")
input_review = st.text_area(
    "Review Text:",
    height=150,
    placeholder="Type your review here... (e.g., This product is ...!)"
)


stopword = set(stopwords.words('english')) - {'not', 'no', 'nor', 'never', 'but', 'very'}
ps = PorterStemmer()

# for preprocessing the text
def transform_text(text):
  text = text.lower() # make lower
  text = nltk.word_tokenize(text) # word tokenize
  y = []
  for i in text:
    if i.isalnum(): #alphanumeric character
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopword and i not in string.punctuation: #remove stopword and punctuations
      y.append(i)
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i)) #stemming

  return " ".join(y)

if st.button("Predict"):
    if input_review.strip() == "":
        st.warning("Please enter a review first!")
    else:
        transformed_review = [transform_text(input_review)]
        result = model.predict(transformed_review)[0]
        
        if result == 1:
            st.success("Positive Review")
        else:
            st.error("Negative Review")