import streamlit as st
import joblib
from time import sleep

# MUST be the first Streamlit command
st.set_page_config(page_title="Fake News Detector",
                   page_icon="📰", layout="centered")

# Load model and vectorizer


@st.cache_resource
def load_model():
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('model.pkl')
    return vectorizer, model


vectorizer, model = load_model()

# Sidebar info
st.sidebar.title("About")
st.sidebar.info(
    "This app detects whether a news article is real or fake using a trained machine learning model."
)

# Title and description
st.title("📰 Fake News Detection")
st.markdown("Enter the text of a news article below and click **Check News** to determine if it's **Real** or **Fake**.")

# Input area
news_input = st.text_area("News Article Text", height=250,
                          placeholder="Paste the news article here...")

# Button with spinner
if st.button("🔍 Check News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text before checking.")
    else:
        with st.spinner("Analyzing the news..."):
            sleep(1)  # Simulate processing delay
            transformed_input = vectorizer.transform([news_input])
            prediction = model.predict(transformed_input)

        if prediction[0] == 1:
            st.success("✅ The news article is **REAL**.", icon="✅")
        else:
            st.error("❌ The news article is **FAKE**.", icon="🚨")
else:
    st.info("📝 Enter some text and press **Check News** to get a result.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Fake News Detector App • Built with Streamlit</p>",
            unsafe_allow_html=True)
