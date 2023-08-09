import streamlit as st
import numpy as np
from PIL import Image
from img_color_gan.pipeline.pipeline_prediction import PredictionPipeline

def main():
    prediction_pipeline = PredictionPipeline("/workspace/Image-Colorization/artifacts/models/model_ckpt.pt")
    st.title('Image Colorizer')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        image = np.array(uploaded_file)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image = Image.open(uploaded_file).convert("RGB")
        width, height = image.size
        
        image = prediction_pipeline.predict(image)
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize((width, height))
        with col2:
            st.image(image, caption="Colored Image", use_column_width=True)

if __name__ == "__main__":
    main()