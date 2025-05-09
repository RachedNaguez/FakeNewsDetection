import streamlit as st
import requests
import json
import joblib
from time import sleep

# MUST be the first Streamlit command
st.set_page_config(page_title="Fake News Detector",
                   page_icon="📰", layout="centered")

# Load vectorizer


@st.cache_resource
def load_vectorizer():
    vectorizer = joblib.load('vectorizer.pkl')
    return vectorizer


vectorizer = load_vectorizer()

# Sidebar info
st.sidebar.title("About")
st.sidebar.info(
    "This app detects whether a news article is real or fake using a trained machine learning model deployed with MLFlow."
)

# Title and description
st.title("📰 Fake News Detection")
st.markdown("Enter the text of a news article below and click **Check News** to determine if it's **Real** or **Fake**.")

# Input area
news_input = st.text_area("News Article Text", height=250,
                          placeholder="Paste the news article here...")

# Function to call the deployed model


def predict_with_deployed_model(text, vectorizer):
    # The URL should match what's printed in the run_deployment.py output
    prediction_url = "http://127.0.0.1:8000/invocations"

    # First, vectorize the text
    vectorized_text = vectorizer.transform([text])

    # Convert to array format
    vectorized_array = vectorized_text.toarray()

    # Format the request as expected by MLflow serving
    headers = {"Content-Type": "application/json"}
    request_data = {
        "inputs": vectorized_array.tolist()
    }

    try:
        response = requests.post(
            url=prediction_url,
            headers=headers,
            data=json.dumps(request_data),
            timeout=10
        )

        if response.status_code == 200:
            # Parse the prediction result
            prediction_result = json.loads(response.text)

            # Handle different response formats
            if isinstance(prediction_result, list):
                return prediction_result[0]
            elif isinstance(prediction_result, dict):
                if "predictions" in prediction_result:
                    return prediction_result["predictions"][0]
                elif "result" in prediction_result:
                    return prediction_result["result"]
                else:
                    return list(prediction_result.values())[0]
            else:
                return prediction_result
        else:
            st.error(
                f"Error calling model API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to model API: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        return None


# Button with spinner
if st.button("🔍 Check News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text before checking.")
    else:
        with st.spinner("Analyzing the news..."):
            sleep(1)  # Simulate processing delay

            # Vectorize and predict using the deployed model
            prediction = predict_with_deployed_model(news_input, vectorizer)

        if prediction is not None:
            try:
                # Convert to int if it's a string or float
                if isinstance(prediction, str) and prediction.strip().isdigit():
                    prediction = int(prediction)
                elif isinstance(prediction, float):
                    prediction = int(prediction)

                if prediction == 1:
                    st.success("✅ The news article is **REAL**.", icon="✅")
                else:
                    st.error("❌ The news article is **FAKE**.", icon="🚨")
            except (ValueError, TypeError) as e:
                st.error(
                    f"Could not interpret prediction result: {prediction}")
                st.error(f"Error: {str(e)}")
else:
    st.info("📝 Enter some text and press **Check News** to get a result.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Fake News Detector App • Built with Streamlit</p>",
            unsafe_allow_html=True)
