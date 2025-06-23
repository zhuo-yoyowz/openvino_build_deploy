import logging
from pathlib import Path
from typing import List

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.react_multimodal.step import MultimodalReActAgentWorker
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    MessageRole,
    LLMMetadata,
)

from llama_index.core.agent.types import (
    BaseAgentWorker,
    Task,
    TaskStep,
    TaskStepOutput,
)


from openvino_genai import VLMPipeline

# --- Wrapper to make VLMPipeline compatible with LLM interface ---
class VLMWrapper:
    def __init__(self, pipeline: VLMPipeline):
        self.pipeline = pipeline

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        prompt = "\n".join([f"{msg.role.value}: {msg.content}" for msg in messages])
        result = self.pipeline.generate(prompt)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=result))

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        result = self.pipeline.generate(prompt)
        return CompletionResponse(text=result)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=512,
            is_chat_model=True,
            is_function_calling_model=False,
        )

# --- Load OpenVINO Vision-Language Model ---
vlm_pipeline = VLMPipeline(
    models_path=Path("model/ov_phi35_vision"),
    device="GPU"
)
llm = VLMWrapper(vlm_pipeline)

# --- Load OpenVINO Embedding Model ---
Settings.embed_model = OpenVINOEmbedding(
    model_id_or_path=str(Path("model/bge-large-FP32").resolve()),
    device="GPU"
)

# --- Load Documents and Create Vector Index ---
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()
query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)

# --- Define Tools for the Agent ---
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools.types import ToolMetadata

tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="retrieval_tool",
            description="Tool for retrieving facts from documents"
        )
    )
]


# --- Initialize Multimodal ReAct Agent ---
agent = MultimodalReActAgentWorker(
    tools=tools,
    multi_modal_llm=llm,
    verbose=True
)

# --- Run a Sample Query with Image ---
image_path = "cat.png"  # Replace with your image file
question = "What is shown in this image?"

from llama_index.core.schema import ImageDocument

# Load the image into an ImageDocument
image_doc = ImageDocument(uri=image_path)

from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer

task = Task(
    task_id="demo-task",
    input=question,
    extra_state={"image_docs": [image_doc]},
    memory=ChatMemoryBuffer.from_defaults()
)


# Initialize the first reasoning step
step = agent.initialize_step(task)

response = agent.run_step(step=step, task=task)


print("Agent response:", response)
