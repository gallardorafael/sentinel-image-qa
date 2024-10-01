import logging

import streamlit as st
import streamlit_cropper
from PIL import Image

from vlm import HF_Qwen2_Chatbot

logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="centered")


def _recommended_box2(img: Image, aspect_ratio: tuple) -> dict:
    width, height = img.size
    return {
        "left": int(10),
        "top": int(10),
        "width": int(width - 10),
        "height": int(height - 10),
    }


@st.cache_resource
def init_llm():
    llm = HF_Qwen2_Chatbot()
    return llm


@st.cache_data
def load_assets():
    user_avatar = Image.open("assets/avatars/user.png")
    assistant_avatar = Image.open("assets/avatars/assistant.png")
    sentinel_logo = Image.open("assets/sentinel_logo_white.png")

    return user_avatar, assistant_avatar, sentinel_logo


class ImageQA_GUI:
    def __init__(self) -> None:
        streamlit_cropper._recommended_box = _recommended_box2
        self.llm = init_llm()
        self.user_avatar, self.assistant_avatar, self.logo = load_assets()

        self.init_sidebar()
        self.init_main()

    def init_sidebar(self):
        # logo
        st.sidebar.image(self.logo, use_column_width=True)

        # file uploader
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        st.session_state["input_image"] = st.sidebar.file_uploader(
            "Chose image to analyze...",
            type="jpeg",
            key=f"uploader_{st.session_state.uploader_key}",
        )

        # reset button
        if st.sidebar.button("Reset history"):
            self.llm.reset_chat_context()
            st.session_state["input_image"] = None
            st.session_state["pil_image"] = None
            st.session_state["messages"] = []
            self._update_uploader_key()

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
                Local assistant for image understanding and visual question answering tasks.
            </div>
            """,
            unsafe_allow_html=True,
        )
        # chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # this returns a file-like object
        if st.session_state["input_image"] is not None:
            # Get a cropped image from the frontend
            uploaded_img = Image.open(st.session_state["input_image"])
            width, height = uploaded_img.size

            new_width = 800
            new_height = int((new_width / width) * height)
            uploaded_img = uploaded_img.resize((new_width, new_height))

            st.session_state["pil_image"] = uploaded_img
            st.session_state.messages.append({"role": "user", "content": uploaded_img})
            # update uploader key to avoid multiple messages
            self._update_uploader_key()

        # showing all the messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message["avatar"]):
                if isinstance(message["content"], Image.Image):
                    st.image(message["content"])
                elif isinstance(message["content"], str):
                    st.markdown(message["content"])

        if prompt := st.chat_input("Ask anything"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt, "avatar": self.user_avatar}
            )
            with st.chat_message("user", avatar=self.user_avatar):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar=self.assistant_avatar):
                response = self.llm.chat(prompt, image=st.session_state.get("pil_image"))
                st.write(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "avatar": self.assistant_avatar}
            )

    def _update_uploader_key(self):
        st.session_state.uploader_key += 1


if __name__ == "__main__":
    interface = ImageQA_GUI()
