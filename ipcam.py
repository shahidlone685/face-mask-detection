import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
import cv2
import tempfile
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model("mask.h5",compile=False)
st.title("Face Mask Detection Using IP Camera")
choice=st.sidebar.selectbox("MY MENU",("HOME","IP CAMERA"))
if(choice=="HOME"):
    st.header("WELCOME🍁")
    st.image("https://png.pngtree.com/thumb_back/fh260/background/20230527/pngtree-white-ip-camera-on-a-white-background-image_2650019.jpg")
elif(choice=="IP CAMERA"):
    k=st.text_input("Enter camera URL (e.g., http://192.168.1.100:8080/video)")
    if st.button("Open Camera"):
        vid = cv2.VideoCapture(k)
        if not vid.isOpened():
            st.error("Failed to open camera stream!")
        else:
            stop_button = st.button("Stop Camera")
            window = st.empty()
            i = 1
            while vid.isOpened():
                # Check if stop button was pressed
                if stop_button:
                    vid.release()
                    st.success("Camera released!")
                    break
                flag, frame = vid.read()
                if not flag:
                    st.warning("Can't receive frame. Stream ended?")
                    break
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face:
                    crop_face1=frame[y:y+w,x:x+l]
                    cv2.imwrite('temp.jpg',crop_face1)
                    crop_face=load_img('temp.jpg',target_size=(150,150,3))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face,axis=0)
                    pred=maskmodel.predict(crop_face)[0][0]
                    if(pred==1):
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                        path="C:/myproject/data/"+str(i)+".jpg"
                        cv2.imwrite(path,crop_face1)
                        i=i+1
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)
                window.image(frame,channels="BGR")  
         
