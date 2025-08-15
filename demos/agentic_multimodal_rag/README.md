# Multimodal RAG for video analytics with LlamaIndex

Constructing a RAG pipeline for text is relatively straightforward, thanks to the tools developed for parsing, indexing, and retrieving text data. However, adapting RAG models for video content presents a greater challenge. Videos combine visual, auditory, and textual elements, requiring more processing power and sophisticated video pipelines.

To build a truly multimodal search for videos, you need to work with different modalities of a video like spoken content, visual. In this notebook, we showcase a Multimodal RAG pipeline designed for video analytics. It utilizes Whisper model to convert spoken content to text, CLIP model to generate multimodal embeddings, and Vision Language model (VLM) to process retrieved images and text messages. The following picture illustrates how this pipeline is working.

![image](https://github.com/user-attachments/assets/a8ebf3fc-7a34-416b-b744-609965792744)

## Contents
The tutorial consists from following steps:
1) Run the FastMCP servers from the command-line 
- CD into the folder where your mcp_server.py, search_mcp_server.py and video_mcp_server.py live:
- cd path\to\your\directory
- Start the retail/cart server (no --reload, to avoid that Windows path-scanning error) with the following command:
  uvicorn mcp_server:app --host 127.0.0.1 --port 8000
- In a second terminal, start the DuckDuckGo search server likewise:
  uvicorn search_mcp_server:app --host 127.0.0.1 --port 8001
- In a third terminal, start the video search based Q&A server likewise:
  uvicorn video_mcp_server:app --host 127.0.0.1 --port 8002
  
2) Then run main.py

Install all dependencies from requirements.txt. Note, all model downloading and converting have to be manually done through "optimum-cli" or downloading from OpenVINO Model Zone from HF directly. Haven't included the model conversion file yet. 

Models used:
- LLM: Qwen2.5-7b-instruct-int4-ov
- VLM: Phi3.5-vision-instruct-int4-ov
- MM embedding: CLIP-VIT-B-32-laion-s34B-b79K
- ASR:distil-whisper-large-v3-int8-ov


In this demonstration, you'll create interactive Q&A system that can answer questions about provided video's content.

## Installation instructions
This is a self-contained example that relies solely on its own code.</br>
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/multimodal-rag/README.md" />

