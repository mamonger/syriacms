import streamlit as st
from PIL import Image
from kraken import binarization, pageseg, recognition

# Load Kraken OCR model
@st.cache_resource
def load_model():
    from kraken.lib.models import load_any
    return load_any("default")  # Replace 'default' with your custom model if applicable

# Streamlit App
def main():
    st.title("Kraken OCR with Streamlit")
    
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "tif"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image with Kraken
        st.write("Running Kraken OCR...")
        model = load_model()
        
        with st.spinner("Processing..."):
            # Step 1: Binarize the image
            bin_img = binarization.nlbin(image)

            # Step 2: Perform page segmentation
            segmentation = pageseg.segment(bin_img)

            # Step 3: Recognize text
            results = recognition.run_ocr(
                model=model,
                im=bin_img,
                bounds=segmentation,
                records=True
            )
        
        # Extract text from results
        extracted_text = "\n".join([line["text"] for line in results])
        st.success("OCR Complete!")
        
        st.subheader("Extracted Text")
        st.text_area("Recognized Text", extracted_text, height=300)
        
        st.download_button(
            label="Download Text",
            data=extracted_text,
            file_name="ocr_output.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
