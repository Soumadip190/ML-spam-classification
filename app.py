import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Auto-download NLTK data (needed for Streamlit Cloud)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.markdown("Enter any SMS message below to check if it's spam.")
st.divider()

def trans_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

input_sms = st.text_area("Enter the message", height=150)

if st.button("🔍 Classify Message", use_container_width=True, type="primary"):
    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_sms = trans_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 **SPAM** — This message looks like spam!")
        else:
            st.success("✅ **NOT SPAM** — This message looks legitimate.")