import streamlit as st
import json
import requests
from PIL import Image
import os

#option 2
import numpy as np
from tensorflow.keras.models import load_model
import cv2



def load_image(image):
	img = Image.open(image)
	return img

def save_uploadedfile(uploadedfile):
     with open(os.path.join("images/img",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
         uploaded_location = os.path.join("images/img",uploadedfile.name)
     return uploaded_location#st.success("Saved File:{} to {}".format(uploadedfile.name, uploaded_location))

def image_predict (image_file):
    model_path = 'application/models/Tumor_VGG_model.h5'  
    h5_model = load_model(model_path)
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)) 
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)
    h5_prediction = h5_model.predict(image)  
    print('Prediction from h5 model: {}'.format(h5_prediction))
    print(h5_prediction)
    probability = h5_prediction[0]
    print("H5 Predictions:")
    print (probability)
    if probability[0] > 0.8:
        brain_tumor_pred = str('%.2f' % (probability[0] * 100) + '% Brain Tumour-Present')
        probability = (probability[0] * 100)
    else:
        brain_tumor_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% No Brain Tumour')
        probability = ((1 - probability[0]) * 100)
    return  brain_tumor_pred
         

st.title("Heart Disease Diagnosis Using MRI Images")



col1, col2 = st.columns([6, 3], gap="medium")

with col1:

        image = st.file_uploader("Upload CT Scan", type=["png","jpg","jpeg"])



        if image is not None:
            # To See details
            file_details = {"filename":image.name, "filetype":image.type,
                "filesize":image.size}
            st.write(file_details)

            #View Uploaded Image
            st.image(load_image(image),width=250)
            #save image to disk
            saved = save_uploadedfile(image)

            #OPTION 1 - with F-API..
            #if st.button ('Analyze'):
                #test_file = open(os.path.join("images/img", image.name), "rb")
                #response = requests.post('http://127.0.0.1:8000/predict/image', files={'file': test_file })
                #prediction = response.json()##json_object["prediction"]
                #st.write(prediction)
                #st. subheader (f"Response from BrainTumor Analyzer API = {prediction}")


            #OPTION 2 - NON API..
            if st.button ('Analyze'):
                with st.spinner('Analyzing...'):
                    prediction = image_predict(saved)
                    #st.write(prediction)
                    st. subheader (f"Image Prediction = {prediction}")
                    st.success(f"Image Prediction = {prediction}", icon="✅")

with col2:
#with st.sidebar:    
    st.write("Developed by AI & IOT Lab https://iot.neu.edu.tr by Mercel Vubangsi (mercel.vubangsi(at)neu.edu.tr)  ")
    #st.header("Sample CT Image")
    st.image( "./heart.png", width=300, caption='Sample Heart MRI Image')#width=400



#streamlit run app.py
#RUN BOTH for F-API...
#uvicorn application.server.main:app

#if __name__ == "__main__":
    #import uvicorn
    #uvicorn.run("application.server.main:app", host="0.0.0.0", port=8000, reload=False, log_level="debug", workers=1, limit_concurrency=1, limit_max_requests=1)











