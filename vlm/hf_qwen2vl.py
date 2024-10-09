import logging
import re
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

from .chat_utils import Qwen2ChatMemoryBuffer
from .defaults import DEFAULT_MODEL_NAME

logging.basicConfig(level=logging.INFO)


class HF_Qwen2_VLM:
    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_MODEL_NAME,
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 1280 * 28 * 28,
        max_tokens: Optional[int] = 256,
        quantize: Optional[bool] = True,
    ) -> None:
        """Initialize the model.

        Args:
            model_name (Optional[str], optional): Model name. Defaults to DEFAULT_MODEL_NAME.
            min_pixels (Optional[int], optional): Minimum number of pixels. Defaults to 256 * 28 * 28.
            max_pixels (Optional[int], optional): Maximum number of pixels. Defaults to 1280 * 28 * 28.
            max_tokens (Optional[int], optional): Maximum number of tokens. Defaults to 256.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        self.max_tokens = max_tokens
        self.quantize = quantize

        self._init_model(model_name, min_pixels, max_pixels)

    def _init_model(self, model_name: str, min_pixels, max_pixels) -> None:
        """Initialize the model and processor.

        Args:
            model_name (str): Model name.
            min_pixels (int): Minimum number of pixels.
            max_pixels (int): Maximum number of pixels.

        Raises:
            Exception: If the model or processor could not be loaded.
        """
        # initializing the model
        try:
            quantization_config = None
            if self.quantize:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", quantization_config=quantization_config
            )
            logging.info(f"Model {model_name} loaded.")
        except Exception as e:
            raise Exception(f"Could not load model {model_name}. Error: {e}")

        # initializing the processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name, min_pixels=min_pixels, max_pixels=max_pixels
            )
        except Exception as e:
            raise Exception(f"Could not load processor for model {model_name}. Error: {e}")

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Generate text based on an image and a prompt.

        Args:
            messages (List[Dict[str, Any]]): Messages, must be in the format of the Qwen2VL model.
                e.g. messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": filepath, PIL image, URL,
                            },
                            {
                                "type": "text",
                                "text": "Describe this image.",
                            },
                        ],
                    },
                ]
        Returns:
            str: Generated text.
        """
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # inference
        generated_ids = self.vlm.generate(**inputs, max_new_tokens=self.max_tokens)

        # trimming the output, generated_ids = image/video tokens + prompt tokens + generated tokens
        # we only want the generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text


class HF_Qwen2_Chatbot(HF_Qwen2_VLM):
    def __init__(
        self,
        model_name: Optional[str] = DEFAULT_MODEL_NAME,
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 1280 * 28 * 28,
        max_tokens: Optional[int] = 1024,
        max_context_tokens: Optional[int] = 2056,
        quantize: Optional[bool] = True,
    ) -> None:
        """
        Initialize a chatbot based on the Qwen2-VL multimodal model. The only attribute of this
        class is the chat context, which is a basic memory buffer implementation.
        Args:
            model_name (Optional[str], optional): Model name. Defaults to DEFAULT_MODEL_NAME.
            min_pixels (Optional[int], optional): Minimum number of pixels. Defaults to 256 * 28 * 28.
            max_pixels (Optional[int], optional): Maximum number of pixels. Defaults to 1280 * 28 * 28.
            max_tokens (Optional[int], optional): Maximum number of tokens. Defaults to 256.
        """
        super().__init__(model_name, min_pixels, max_pixels, max_tokens, quantize)

        self.chat_context = Qwen2ChatMemoryBuffer(max_context_tokens=max_context_tokens)

    def chat(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        format_latex: Optional[bool] = True,
    ) -> str:
        """Chat with the assistant. The assistant will generate a response based on the prompt, the
        context image and the chat context in self.chat_context.

        Args:
            prompt (str): User prompt.
            image (Optional[Union[str, Image.Image]], optional): Image. Defaults to None.
        Returns:
            str: Assistant response.
        """
        # if only an image was received, update the image context
        if image is not None:
            logging.info("Updating image context.")
            self.chat_context.update_image_context(prompt, image)

        # add the message to the chat context
        self.chat_context.add_user_message(prompt)

        # generate the response based on the prompt and the chat context
        formatted_chat_context = self.chat_context.get_chat_context()
        response = self.generate(messages=formatted_chat_context)

        if format_latex:
            response = self._format_equations_to_latex(response)

        logging.info(f"Chat context: {formatted_chat_context}")

        # adding the response to the chat context
        self.chat_context.add_assistant_message(response)

        return response

    def reset_chat_context(self):
        """Reset the chat context.

        Use this function to clear the chat history and start a fresh conversation.
        """
        self.chat_context.reset_context()

    def _format_equations_to_latex(self, response: str) -> str:
        """Qwen2-VL outputs for normal text, lists and code, but Streamlit fails to parse the math
        (latex) outputs correctly, so this function replaces:

        - substrings in the format \\(...\\) with $...$
        - substrings in the format \\[...\\] with $...$

        Args:
            strings (List[str]): List of strings to process.
        Returns:
            The processed string.
        """
        response = re.sub(r"\\\[", "$", response)
        response = re.sub(r"\\\]", "$", response)
        response = re.sub(r"\\\(", "$", response)
        response = re.sub(r"\\\)", "$", response)

        return response
