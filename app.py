# app.py
import streamlit as st
from PIL import Image
import os
from objectDetection.pipeline.prediction import run_object_detection

st.title("Object Detection App")

# Step 1: Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Step 2: Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Save the uploaded image temporarily
    temp_file_path = os.path.join("temp_image.png")
    image.save(temp_file_path)

    # Step 3: Button to trigger Object Detection
    if st.button('Run Object Detection'):
        # Run object detection using the predict.py script
        st.write("Running object detection...")

        # Call the object detection function
        annotated_image, detection_results = run_object_detection(temp_file_path)

        # Step 4: Display Annotated Image
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
        st.write("Object detection completed!")
        for i in detection_results:
            st.write(i)

        # Remove temporary image file
        os.remove(temp_file_path)