pip install transformers streamlit

import streamlit as st
# Load model
processor = AutoImageProcessor.from_pretrained("ttangmo24/vit-base-classification-Eye-Diseases")
model = AutoModelForImageClassification.from_pretrained("ttangmo24/vit-base-classification-Eye-Diseases")

# Streamlit app
def main():
    st.title("Image Classification with Hugging Face")
    st.write("Upload an image for classification.")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Process uploaded image
        input_image = processor(uploaded_file)
        
        # Perform classification
        with st.spinner('Classifying...'):
            outputs = model(input_image.pixel_values)
            predicted_class_idx = outputs.logits.argmax().item()

        # Display results
        st.write("Predicted Class:", predicted_class_idx)

if __name__ == "__main__":
    main()
