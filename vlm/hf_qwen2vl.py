import logging
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .prompts import DEFAULT_CHAT_PROMPT, DEFAULT_NO_IMAGE_RESPONSE

logging.basicConfig(level=logging.INFO)


class HF_Qwen2_VLM:
    def __init__(
        self,
        model_name: Optional[str] = "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 1280 * 28 * 28,
        max_tokens: Optional[int] = 256,
    ) -> None:
        """Initialize the model.

        Args:
            model_name (Optional[str], optional): Model name. Defaults to "Qwen/Qwen2-VL-2B-Instruct".
            min_pixels (Optional[int], optional): Minimum number of pixels. Defaults to 256 * 28 * 28.
            max_pixels (Optional[int], optional): Maximum number of pixels. Defaults to 1280 * 28 * 28.
            max_tokens (Optional[int], optional): Maximum number of tokens. Defaults to 256.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

        self.max_tokens = max_tokens

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
            self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
            )
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
        model_name: Optional[str] = "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 1280 * 28 * 28,
        max_tokens: Optional[int] = 256,
        force_image: Optional[bool] = False,
    ) -> None:
        super().__init__(model_name, min_pixels, max_pixels, max_tokens)

        self.system_prompt = DEFAULT_CHAT_PROMPT
        self.chat_context: List[Dict[str, Any]] = self._init_system_prompt()
        self.is_image_set = False
        self.force_image = force_image

    def chat(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        return_stream: Optional[bool] = True,
    ) -> str:
        """"""
        # return a message indicating that the image is required
        if self.force_image:
            if image is None and not self.is_image_set:
                return DEFAULT_NO_IMAGE_RESPONSE

        # add the image to the context if it is not set
        if image is not None and not self.is_image_set:
            self._add_image_to_context(image, prompt)
            self.is_image_set = True
            return self.generate(messages=self.chat_context)

        # generate the message
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }

        # generate the response based on the prompt and the chat context
        self.chat_context.append(message)
        response = self.generate(messages=self.chat_context)

        # adding the response to the chat context
        self.add_assistant_message(response)

        return response

    def add_assistant_message(self, message: str) -> None:
        """"""
        message = {
            "role": "system",
            "content": message,
        }
        self.chat_context.append(message)

    def _add_image_to_context(self, image: Union[str, Image.Image], prompt: str) -> None:
        """"""
        message = {
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
        self.chat_context.append(message)

    def _init_system_prompt(self):
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ]

    def reset_chat_context(self):
        self.chat_context = self._init_system_prompt()
        self.is_image_set = False
