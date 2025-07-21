import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

print("Libraries are Imported")


model=load_model(r"E:\\Codes\Projects\ML\Leaf_Disease\models\rice_disease_cnn.h5")

class_lables = ['leafscald','leafblast','brownspot','bacterialleafblight','healthy']

remedies={
    'leafscald':"Use Recommended Fungicides Such as Benomyl and Improve field Drainage.",
    'leafblast':"Apply Fungicides Like Tricyclazole And Avoid Excess Nitrogen Fertilizers.",
    'brownspot':"Use Resistant Varieties,Balanced Fertilization, and Fungicide Spraying If Needed.",
    'bacterialleafblight':"Use Disease-Free Seeds,Proper Water Management, and Copper-Based Bactericides.",
    'healthy' :" The Plant Appears Healthy ! Continue Regular Monitoring."
}

st.title("Rice Plant Disease Detector - By Using CNN Model")

uploaded_file = st.file_uploader("Upload An Image Of Rice Leaf",type=['jpg','png','jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_container_width=True)


    img= img.resize((128,128))
    img_array=image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0)

    prediction=model.predict(img_array)
    predicted_class_index=np.argmax(prediction)
    predicted_class=class_lables[predicted_class_index]
    confidence= prediction[0][predicted_class_index]*100

    st.subheader(f"Predicted Disease:**{predicted_class}")
    st.subheader(f"**Confidence:** {confidence:.2f}%")
    st.info(f"**Suggested Remedy:**{remedies[predicted_class]}")