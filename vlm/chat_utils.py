import logging
import time
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from .prompts import DEFAULT_CHAT_PROMPT


class Qwen2ChatMemoryBuffer:
    def __init__(self, max_context_tokens: int) -> None:
        """Initialize the chat memory buffer.

        This buffer stores the chat context and system prompts, each message contains the
        role of the message and the content. Currently, only 1 image message is allowed.

        Args:
            max_context_tokens (Optional[int], optional): Maximum number of tokens. If the number of
                tokens exceeds this value, the context will be reset.
              Defaults to 1024.
        """
        self.system_prompt: Dict[str, Any] = ""
        self.chat_context: List[Dict[str, Any]] = []
        self.max_context_tokens = max_context_tokens

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
        # reset the chat context to avoid the previous messages overweighing the image
        self.chat_context = []

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
        # flushing old tokens if needed
        self._flush_text_context()

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

        Returns:
            bool: True if the context was reset successfully, otherwise False.
        """
        try:
            self.chat_context: List[Dict[str, Any]] = []
            self.image_message: Dict[str, Any] = None
            return True
        except Exception:
            return False

    def _flush_text_context(self):
        """This function flushes only the text context, the system prompt and image context will be
        kept.

        Returns:
            bool: True if the context was reset successfully, otherwise False.
        """
        try:
            # search for all content keys in the chat context, only with type==text
            # until self.max_context_tokens (words) are found and return the index of that element
            index = 0
            tokens = 0
            for i, message in enumerate(self.chat_context):
                for content in message["content"]:
                    if content["type"] == "text":
                        tokens += len(content["text"].split())
                        if tokens >= self.max_context_tokens:
                            index = i
                            break
            if index > 0:
                logging.info(f"Flushing {tokens} context tokens until message {index}")

            # removing all messages until index
            self.chat_context = self.chat_context[index:]

            return True
        except Exception:
            return False


def simulate_stream(static_response: str, delay: Optional[float] = 0.02):
    """Simulate a stream of text.

    Args:
        static_response (str): Static string response to be returned.

    Yields:
        str: Word by word.
    """
    for word in static_response.split(" "):
        yield word + " "
        time.sleep(delay)
