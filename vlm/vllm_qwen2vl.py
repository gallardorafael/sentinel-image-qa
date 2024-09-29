from PIL import Image
from typing import Dict, Optional, Union
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

class vLLM_Qwen2_VL:
    def __init__(self, 
                 model_name: Optional[str] = "Qwen/Qwen2-VL-2B-Instruct",
                 sampling_params: Optional[dict] = {
                     "temperature": 0.1, 
                     "top_p": 0.001, 
                     "repetition_penalty": 1.05, 
                     "max_tokens": 256,
                     "stop_token_ids": []},
                 limits: Optional[Dict[str, int]] = {"image": 10, "video": 10},
                 min_pixels: Optional[int] = 256*28*28,
                 max_pixels: Optional[int] = 1280*28*28) -> None:
        """
        Wrapper for Qwen2-VL vision-language model, over vLLM.

        Args:
            model_name (str, optional): Model name to use. Defaults to "Qwen/Qwen2-VL-2B-Instruct".
            sampling_params (dict, optional): Sampling parameters. Defaults to {
                "temperature": 0.1, 
                "top_p": 0.001, 
                "repetition_penalty": 1.05, 
                "max_tokens": 256,
                "stop_token_ids": []}.
            limits (dict, optional): Limits for multimodal data. Defaults to {"image": 10, "video": 10}.
            min_pixels (int, optional): Minimum number of pixels for image. Defaults to 256*28*28.
            max_pixels (int, optional): Maximum number of pixels for image. Defaults to 1280*28*28.
        """
        
        # note: VLLM is not ready to work with Qwen2-VL model because of a config mismatch
        # in the rope_scaling` for 'rope_type'='default': {'mrope_section'}
        # Error we are getting: https://github.com/vllm-project/vllm/pull/7905#issuecomment-2352109899
        # Transformers conversation: https://github.com/huggingface/transformers/issues/33401
        # The PR that should fix this is: https://github.com/huggingface/transformers/pull/33753
        # Once Transformers solve this, VLLM will need to update.
        self.llm = LLM(
            model=model_name,
            limit_mm_per_prompt=limits,
        )

        self.sampling_params = SamplingParams(**sampling_params)

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, image: Union[Image.Image, str], prompt: str) -> str:
        """
        Generate text based on an image and a prompt.
        Args:
            image (Union[Image.Image, str]): Image to use.
            prompt (str): Prompt to use.
        Returns:
            str: Generated text.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {
                    "type": "image",
                    "image": image,

                    # min_pixels & max_pixels are optional
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ]},
        ]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        
        return outputs[0].outputs[0].text