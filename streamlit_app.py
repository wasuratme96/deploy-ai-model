
import streamlit as st
from transformers import AutoModelForImageClassification, pipeline

# Load model
#st.cache()
#def get_processor(model_name:str):
#    return AutoImageProcessor.from_pretrained(model_name)

st.cache()
def get_model(model_name:str):
    return pipeline("image-classification", model=model_name)

# Streamlit app
def main():
    st.title("Image Classification with Hugging Face")
    st.write("Upload an image for classification.")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file)
        with st.spinner('Classifying...'):
            outputs = model(uploaded_file)
            # Display results
            st.write("Predicted Class:", outputs)

if __name__ == "__main__":
    model_name = "ttangmo24/vit-base-classification-Eye-Diseases"
    model = get_model(model_name)
    main()
