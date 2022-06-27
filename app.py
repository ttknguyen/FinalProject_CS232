from distutils.command.upload import upload
import os
from re import sub
import streamlit as st
import streamlit.components.v1 as components
from api_speech2text import speech2text, loadmodel
import tensorflow as tf
import pydub


model = loadmodel('./model')

# Setup page configuration
st.set_page_config(
    page_title="speech2text",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''', unsafe_allow_html=True)

# Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''', unsafe_allow_html=True)

# Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''', unsafe_allow_html=True)  # lightmode

# Hide footer
hide_streamlit_footer = """
    <style>
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }
    </style>
"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)


def save_audio(uploaded_file):
    save_path = 'Temp/audio.wav'
    if uploaded_file is not None:
        if uploaded_file.name.endswith('wav'):
            audio = pydub.AudioSegment.from_wav(uploaded_file)
            file_type = 'wav'
        elif uploaded_file.name.endswith('mp3'):
            audio = pydub.AudioSegment.from_mp3(uploaded_file)
            file_type = 'mp3'

        audio.export(save_path, format=file_type)
    return save_path


def handle_submit(submit, implement_option, uploaded_file):
    error_title = "<h6 style='text-align: center; font-size:15px; color: red;'>No input data</h6>"

    if submit:
        if uploaded_file == None: 
            st.markdown(error_title, unsafe_allow_html=True)
            return
        
        audio_path = save_audio(uploaded_file)
        result = None
        if implement_option == 'SpeechRecognition Model':
            result = speech2text(model, audio_path, 'ASR_lib')

        if implement_option == 'Custom Model':
            result = speech2text(model, audio_path)

        st.header("RESULT:")
        st.write(result)

        _, _, _, _, col, _, _, _, _ = st.columns([1]*8+[1.18])
        col.button('CLEAR')
        return

def main_page(input_option, implement_option):
    page_title = "<h1 style='text-align: center; font-family:Impact; font-size:100px; color: #00BFFF;'> SPEECH TO TEXT </h1>"

    # WARNING: RECORD AUDIO IN PROGRESS OF DEVELOPMENT, NOT READY TO USE
    if input_option == 'Record Audio':
        st.markdown(page_title, unsafe_allow_html=True) # Write title

        title_record = "<h2 style='text-align: center; font-size:20px; color: #000000;'>Record your audio here</h2>"
        st.markdown(title_record, unsafe_allow_html=True)

        st_audiorec = components.declare_component("st_audiorec", path="Components/st_audiorec/frontend/build")
        st_audiorec()

        # Create & Align center submit button
        _, _, _, _, col, _, _, _, _ = st.columns([1]*8+[1.18])
        submit = col.button("SUBMIT")
    
    else:
        st.markdown(page_title, unsafe_allow_html=True) # Write title

        title_upload = "<h2 style='text-align: center; font-size:20px; color: #000000;'>Upload your audio file here</h2>"
        st.markdown(title_upload, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False, type=['mp3', 'wav', 'ogg', 'wma', 'aiff'])

        # Create & Align center submit button
        _, _, _, _, col, _, _, _, _ = st.columns([1]*8+[1.18])
        submit = col.button("SUBMIT")

        handle_submit(submit, implement_option, uploaded_file)


def audio_record():
    # Create sidebar title
    sidebar_title = "<h1 style='text-align: center; font-family:Arial Black; font-size:40px; color: #000000;'><b>WELCOME TO MY PAGE</b></h1></br>"
    st.sidebar.markdown(sidebar_title, unsafe_allow_html=True)

    # Create option box for selecting method
    input_option = st.sidebar.selectbox('Choose input method:', ('Upload Audio', 'Record Audio'))
    implement_option = st.sidebar.selectbox('Choose implement method:', ('SpeechRecognition Model', 'Custom Model'))
    
    main_page(input_option, implement_option)

    # Info about this project
    sidebar_info = """</br></br></br></br></br></br></br>
                    <h3 style='text-align: left; font-family:Arial; font-size:20px; color: #000000;'>💡About my project</h3>
                    <h6 style='text-align: left; font-family:Arial narrow; font-size:20px; color: gray;'>
                        ⫭ Subject: </br> Introduction to Multimedia Computing</h6>
                    <h6 style='text-align: left; font-family:Arial narrow; font-size:20px; color: gray;'>
                        ⫭ Class: </br> CS232.M22.KHCL</h6>
                    <h6 style='text-align: left; font-family:Arial narrow; font-size:20px; color: gray;'>
                        ⫭ Contributors:
                        <ul>
                            <li>Thai Tran Khanh Nguyen &emsp;- 19520188</li>
                            <li>Doan Nguyen Nhat Quang &ensp;- 19520235</li>
                            <li>Nguyen Pham Vinh Nguyen&nbsp;- 19520186</li>
                            <li>Nguyen Khanh Nhu &emsp;&emsp;&emsp;- 19520209</li>
                        </ul>
                    </h6>"""
    st.sidebar.markdown(sidebar_info, unsafe_allow_html=True)


if __name__ == '__main__':
    # call main function
    audio_record()
