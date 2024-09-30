import logging
from typing import Optional, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logging.basicConfig(level=logging.INFO)


class HF_Qwen2_VL:
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

    def generate(self, image: Union[Image.Image, str], prompt: str) -> str:
        """Generate text based on an image and a prompt.

        Args:
            image (Union[Image.Image, str]): Image to use.
            prompt (str): Prompt to use.
        Returns:
            str: Generated text.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant, based on a multimodal vision-language model",
            },
            {
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
            },
        ]

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
