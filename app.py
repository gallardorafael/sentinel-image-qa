import logging

import streamlit as st
import streamlit_cropper
from PIL import Image
from streamlit_cropper import st_cropper

logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="centered")

def _recommended_box2(img: Image, aspect_ratio: tuple) -> dict:
    width, height = img.size
    return {
        "left": int(10),
        "top": int(10),
        "width": int(width - 30),
        "height": int(height - 30),
    }

class ImageQA_GUI:
    def __init__(self):
        streamlit_cropper._recommended_box = _recommended_box2
        
        self.init_sidebar()
        self.init_main()

    def init_sidebar(self):
        # logo
        st.sidebar.image("assets/sentinel_logo_white.png", use_column_width=True)
        
        # file uploader
        st.session_state["input_image"] = st.sidebar.file_uploader("Chose image to analyze...", type="jpeg")
    
    def init_main(self):
        st.markdown(
            """
            <style>
            .title {
                text-align: center;
                font-size: 50px;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .description {
                text-align: center;
                font-size: 18px;
                color: gray;
                margin-bottom: 30px;
            }
            </style>
            <div class="title">Image QA</div>
            <div class="description">
                Local assistant for image understanding and vision question answering tasks.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # this returns a file-like object
        if st.session_state["input_image"] is not None:
            # Get a cropped image from the frontend
            uploaded_img = Image.open(st.session_state["input_image"])
            width, height = uploaded_img.size

            new_width = 800
            new_height = int((new_width / width) * height)
            uploaded_img = uploaded_img.resize((new_width, new_height))

            with st.empty():
                cropped_img = st_cropper(
                    uploaded_img,
                    box_color="#f20604",
                    realtime_update=False,
                    aspect_ratio=(16, 9),
                )
            st.session_state["cropped_img"] = cropped_img

        with st.form("my_form"):
            question = st.text_area("Enter your question:")
            submitted = st.form_submit_button("Submit")

            if question and submitted:
                answer = "I'm sorry, I don't know the answer to that question."

                # Display the question and response in a chatbot-style box
                st.chat_message("user").write(question)
                st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    interface = ImageQA_GUI()