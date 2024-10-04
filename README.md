# SENTINEL V-Companion

SENTINEl V-Companion is a fully local, open-source multi-modal and multi-language assistant that leverages the power of the [Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) vision-language model to solve
traditional text-based problems but also image-based ones, such as image understanding and image reasoning. This project allows the user to interact with a powerful
language model that is able to understand both images and text, without requiring suscriptions, internet connection and without any of your data leaving your computer.

Demo video:

## Set up

With Docker, the set up is as easy as it gets, just clone the repository and then run: `docker compose up -d`

This will pull all the repositories and build the image. This will also launch the Streamlit app, which should be available at `http://localhost:8501/`

After this, you should be able to see the user interface as in the demo video above.

## Usage

Chat with V-Companion as you would do with any other chatbot out there (ChatGPT, Claude, etc.), the answers will be of less quality because we are using a really small model as default (2B parameters), but if you have enough VRAM, you could use the bigger and better versions of Qwen2-VL.

On the left sidebar, you will find two buttons:

1. Browse files: Which will allow you to select 1 image, which will be used as context for the language model. At the moment, V-Companion is only able to analyze 1 image, but you can chat and iterate with multiple questions about that image, or any other topic.
2. Reset history: Use this button if you want to start a fresh conversation with V-Companion, this will reset both the image context and the chat history.

## Performance notes

Test environment:
- os: pop-os 22.04
- inference hardware: NVIDIA GeForce RTX 4060 Laptop GPU

Task 1:
- generating the answer for the prompt "Tell me more about the Vietnam war", max tokens is 1024. No image context.
- Results: 50 tokens (words) per second.

Task 2:
- generating the answer for the prompt "Tell me more about the Vietnam war", max tokens is 1024. With an image context with size 1900x1200 pixeles.
- Results: 32 tokens (words) per second.


## Roadmap for 2.x
Here are some ideas that I would like to implement for the 2.x version of SENTINEL V-Companion:
- Reduced size and complexity of the project's dependencies, such as removing the need of using the Transformers library, and thus PyTorch.
- Implement a way to use ONNX models instead of torch-based models.
- Use bigger models with the same hardware, maybe with quantization or other optimizations.
- Work with videos, so you can chat about the stuff happening in a video and not just static images.
- Improve the chat memory management, the buffer memory was straightforward to implement but it is not the best option.
- Give the chatbot the ability to access larger context information such as PDF documents, web pages or code (RAG).


## Acknowledgments

- Thanks to the [Qwen2-VL](https://qwenlm.github.io/blog/qwen2-vl/) team for all the effort they put into the research and development of this model, it is truly amazing and powerful. Kudos for making this open-source!
- Thanks to [Streamlit.io](https://github.com/streamlit/streamlit) for open-sourcing this great tool. This allowed me to create a stable and good-looking GUI without any formal knowledge on front-end development.