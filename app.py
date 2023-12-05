import streamlit as st
import time
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# generator is trained on images of size (256,256,3)
image_size = 256

# generator model contains both encoder and decoder
# it take is two inputs: a cover image and secret image
# and gives two outputs: a stego-image and secret image extracted from the secret image
# load the model from the path provided here
model_path = './models/Generator_200.h5'
generator = load_model(model_path)


def preprocess_images(cover, secret):
    cover = Image.open(cover).convert("RGB")
    secret = Image.open(secret).convert("RGB")
    # resize the image
    cover = cover.resize((image_size, image_size))  # Adjust the size based on your model's input requirements
    secret = secret.resize((image_size, image_size))
    
    # rescaling the image to [-1, 1]
    cover = (np.array(cover) /255.0) * 2.0 - 1.0    
    secret = (np.array(secret) / 255.0) * 2.0 - 1.0
    
    # add the batch dimension to both the images 
    cover = np.expand_dims(cover, axis=0)  
    secret = np.expand_dims(secret, axis=0)

    return cover, secret

def get_results(cover,secret):
    cover_img, secret_img = preprocess_images(cover,secret)
    
    result = generator.predict([cover_img, secret_img])
    
    # remove the batch dimension form the resultant images
    stego_img = np.squeeze(result[0])
    re_constructed_secret = np.squeeze(result[1])

    # conver the pixel values of the images to [0, 1] inorder to display them in streamlit
    stego_img = (stego_img + 1) / 2.0
    re_constructed_secret = (re_constructed_secret+ 1) / 2.0

    return stego_img, re_constructed_secret



def main():
    st.set_page_config(
        page_title="Secura-Steganography GAN",
        page_icon=":🟢:",
        layout="wide",
        initial_sidebar_state="auto")
    
    hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    

    #for css file backgroud
    with open('./css/style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    
    # st.title(":green[Steganography Using GAN]")
    original_title = '<p style="font-size: 100px;">SECURA</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    
    # st.title(":white[SECURA]")
    st.subheader(" Create steganographic images of your choice")
    

    #sidebar
    

    st.sidebar.title("Menu",help="Upload the images to get started")
    st.sidebar.header("Upload your images to get started.")
    
    st.sidebar.divider()
    exit = st.empty()

    # Add some text or instructions
    # st.sidebar.write("Upload Cover Image and Secret Image ")

    with exit.container():    
        cover = st.sidebar.file_uploader("Cover Image", type=["jpg", "jpeg", "png"])
        secret = st.sidebar.file_uploader("Secret Image", type=["jpg", "jpeg", "png"])
        # Display the uploaded images
        col1, col2 = st.columns(2)
        if (cover is not None) and (secret is not None) :
            with col1:
                st.header("Cover Image")
                st.image(cover, caption="Uploaded Cover Image", width=500) #use_column_width=True,
   
            with col2:
                st.header("Secret Image")
                st.image(secret, use_column_width=500, width=500,caption="Uploaded Secret Image")    
            
    st.divider()
    #stagano creating button
    scol1, scol2 = st.sidebar.columns(2)
    if st.sidebar.button("Generate"):
        if cover is  None or secret is None:
            st.warning("Please upload both Cover and Secret Images")
        else:
            stego, re_constructed_secret = get_results(cover, secret)
            
            with st.container():
                with st.spinner(text=":green[Encoding...]"):
                    time.sleep(1)
                
                st.success("[Done!]")
                col3, col4 = st.columns(2)
                with col3:
                    st.header("Stego Image")
                    st.image(stego,caption= "Stego Image", width= 500)
                    # st.download_button(label="Download image",data=stego,file_name="flower.png",mime="image/png")   => this doesnt work                
                with col4:
                    st.header("Reconstructed Secret Image")
                    st.image(re_constructed_secret,caption= "Reconstructed Secret Image", width= 500)
                
    if st.sidebar.button("Clear"):
        exit.empty()
          
 
    

if __name__ == "__main__":
    main()
