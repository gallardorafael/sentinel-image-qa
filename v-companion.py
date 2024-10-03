import logging
from typing import Tuple

import streamlit as st
from PIL import Image

from vlm import HF_Qwen2_Chatbot
from vlm.chat_utils import simulate_stream

logging.basicConfig(level=logging.INFO)

st.set_page_config(layout="centered")


@st.cache_resource
def init_llm() -> HF_Qwen2_Chatbot:
    """Initialize the Qwen2-VL chatbot, which includes loading the weights into the GPU memory.

    Using st.cache_resource to avoid loading the model multiple times.
    """
    llm = HF_Qwen2_Chatbot()
    return llm


@st.cache_data
def load_assets() -> Tuple[Image.Image, Image.Image, Image.Image]:
    """Load all the assets required for the GUI.

    Using st.cache_data to avoid loading the assets multiple times.
    """
    user_avatar = Image.open("assets/avatars/user.png")
    assistant_avatar = Image.open("assets/avatars/assistant.png")
    sentinel_logo = Image.open("assets/sentinel_logo_white.png")

    return user_avatar, assistant_avatar, sentinel_logo


class VCompanion_GUI:
    def __init__(self) -> None:
        """Initialize the V-Companion GUI, which includes the chatbot interface and the sidebar."""
        self.llm = init_llm()
        self.user_avatar, self.assistant_avatar, self.logo = load_assets()

        self.init_sidebar()
        self.init_main()

    def init_sidebar(self):
        """Sidebar interface for the V-Companion chatbot.

        It includes the file uploader and the reset button.
        """
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
        """Main interface for the V-Companion chatbot.

        It includes the management of the chat history and the interaction with the chatbot.
        """
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
            <div class="title">V-Companion</div>
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
            st.session_state.messages.append(
                {"role": "user", "content": uploaded_img, "avatar": self.user_avatar}
            )
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
                with st.status("", expanded=True):
                    response = self.llm.chat(prompt, image=st.session_state.get("pil_image"))
                    st.write_stream(simulate_stream(response))
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "avatar": self.assistant_avatar}
            )

    def _update_uploader_key(self):
        st.session_state.uploader_key += 1


if __name__ == "__main__":
    """Main entry point for the Streamlit app."""
    interface = VCompanion_GUI()
