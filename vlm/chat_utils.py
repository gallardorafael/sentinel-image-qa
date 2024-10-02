import time
from typing import Any, Dict, List, Union

from PIL import Image

from .prompts import DEFAULT_CHAT_PROMPT


class Qwen2ChatMemoryBuffer:
    def __init__(self) -> None:
        """Initialize the chat memory buffer.

        This buffer stores the chat context and system prompts, each message contains the
        role of the message and the content. Currently, only 1 image message is allowed.
        """
        self.system_prompt: Dict[str, Any] = ""
        self.chat_context: List[Dict[str, Any]] = []

        # intialize the chat context
        self.image_message: Dict[str, Any] = None
        self._add_system_prompt()

    def add_assistant_message(self, message: str):
        """Add message to the chat context with the role of the assistant.

        Args:
            message (str): Assistant message.
        """
        message = {
            "role": "system",
            "content": message,
        }
        self.chat_context.append(message)

    def add_user_message(self, message: str):
        """Add message to the chat context with the role of the user.

        Args:
            message (str): User message.
        """
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
        """Update the image context with the received image and prompt.

        Args:
            prompt (str): Prompt.
            image (Union[str, Image.Image]): Image.
        """
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

    def get_chat_context(self) -> List[Dict[str, Any]]:
        """Format the current chat context. If the image message is not None, the image message
        will be included in the chat context, otherwise, only the system prompt will be included.

        Returns:
            List[Dict[str, Any]]: Formatted chat context.
        """
        if self.image_message is None:
            curent_context = [self.system_prompt] + self.chat_context
        else:
            curent_context = [self.system_prompt, self.image_message] + self.chat_context

        return curent_context

    def _add_system_prompt(self):
        """Add the system prompt to the chat context.

        The system prompt is the default message that indicates the model how to behave.
        """
        message = {
            "role": "system",
            "content": DEFAULT_CHAT_PROMPT,
        }
        self.system_prompt = message

    def reset_context(self) -> bool:
        """Reset the chat context.

        This function clears the chat history and starts a fresh conversation.
        """
        try:
            self.chat_context: List[Dict[str, Any]] = []
            self.image_message: Dict[str, Any] = None
        except Exception:
            return False


def simulate_stream(static_response: str):
    for word in static_response.split(" "):
        yield word + " "
        time.sleep(0.03)
