from typing import Any, Dict, List, Union

from PIL import Image

from .prompts import DEFAULT_CHAT_PROMPT


class Qwen2ChatContext:
    def __init__(self) -> None:
        self.system_prompt: Dict[str, Any] = ""
        self.chat_context: List[Dict[str, Any]] = []

        # intialize the chat context
        self.image_message: Dict[str, Any] = None
        self._add_system_prompt()

    def add_assistant_message(self, message: str):
        message = {
            "role": "system",
            "content": message,
        }
        self.chat_context.append(message)

    def add_user_message(self, message: str):
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message,
                }
            ],
        }
        self.chat_context.append(message)

    def update_image_context(self, prompt: str, image: Union[str, Image.Image]):
        self.image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }

    def get_chat_context(self):
        if self.image_message is None:
            curent_context = [self.system_prompt] + self.chat_context
        else:
            curent_context = [self.system_prompt, self.image_message] + self.chat_context

        return curent_context

    def _add_system_prompt(self):
        message = {
            "role": "system",
            "content": DEFAULT_CHAT_PROMPT,
        }
        self.system_prompt = message

    def reset_context(self):
        self.chat_context: List[Dict[str, Any]] = []
        self.image_message: Dict[str, Any] = None
