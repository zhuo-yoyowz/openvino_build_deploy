from typing import List, Optional, Sequence, Union, Any, Callable
import concurrent.futures
import threading

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse, LLMMetadata
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

# Ensure this import is available
try:
    from openvino_genai import VLMPipeline
except ImportError:
    VLMPipeline = Any  # fallback typing in case openvino_genai is not yet installed

class OpenVINOVLM(LLM):
    """
    LlamaIndex-compatible wrapper for OpenVINO Vision-Language Models (e.g., Phi-3.5 Vision).
    """

    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW)
    max_new_tokens: int = Field(default=DEFAULT_NUM_OUTPUTS)
    generate_kwargs: dict = Field(default_factory=lambda: {"temperature": 0.1, "top_p": 0.95})
    device: str = Field(default="CPU")
    is_chat_model: bool = Field(default=True)

    # Prompt customization
    query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}"
    messages_to_prompt: Optional[Callable] = None
    completion_to_prompt: Optional[Callable] = None

    callback_manager: Optional[CallbackManager] = None
    pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT
    output_parser: Optional[BaseOutputParser] = None

    _vlm = PrivateAttr()

    def __init__(
        self,
        vlm_pipeline: VLMPipeline,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        generate_kwargs: Optional[dict] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            generate_kwargs=generate_kwargs or {"temperature": 0.1, "top_p": 0.95},
            callback_manager=callback_manager,
            is_chat_model=True,
        )
        self._vlm = vlm_pipeline

    def _format_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = ""
        for m in messages:
            if m.role == "system":
                prompt += f"<|system|>\n{m.content}</s>\n"
            elif m.role == "user":
                prompt += f"<|user|>\n{m.content}</s>\n"
            elif m.role == "assistant":
                prompt += f"<|assistant|>\n{m.content}</s>\n"
        prompt += "<|assistant|>\n"
        return prompt

    def _run_with_timeout(self, func, timeout=30, **kwargs):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, **kwargs)
            return future.result(timeout=timeout)

    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> ChatResponse:
        prompt = self._format_prompt(messages)
        image = None
        for m in messages:
            if hasattr(m, "image") and m.image is not None:
                image = m.image
                break
        generate_args = {"prompt": prompt, **self.generate_kwargs}
        if image is not None:
            generate_args["image"] = image
        output = self._run_with_timeout(self._vlm.generate, **generate_args)
        return ChatResponse(message=ChatMessage(role="assistant", content=output["text"]))

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        output = self._run_with_timeout(self._vlm.generate, prompt=prompt, **self.generate_kwargs)
        return CompletionResponse(text=output["text"])

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs):
        prompt = self._format_prompt(messages)
        image = None
        for m in messages:
            if hasattr(m, "image") and m.image is not None:
                image = m.image
                break
        generate_args = {"prompt": prompt, **self.generate_kwargs}
        if image is not None:
            generate_args["image"] = image

        if hasattr(self._vlm, "stream") and callable(getattr(self._vlm, "stream")):
            try:
                for token in self._vlm.stream(**generate_args):
                    yield ChatResponse(message=ChatMessage(role="assistant", content=token))
            except Exception as e:
                yield ChatResponse(message=ChatMessage(role="assistant", content=f"[ERROR] {str(e)}"))
        else:
            output = self._run_with_timeout(self._vlm.generate, **generate_args)
            yield ChatResponse(message=ChatMessage(role="assistant", content=output["text"]))

    def acomplete(self, *args, **kwargs):
        raise NotImplementedError("Async completion not implemented.")

    def achat(self, *args, **kwargs):
        raise NotImplementedError("Async chat not implemented.")

    def astream_chat(self, *args, **kwargs):
        raise NotImplementedError("Async streaming chat not implemented.")

    def astream_complete(self, *args, **kwargs):
        raise NotImplementedError("Async streaming complete not implemented.")

    def stream_complete(self, *args, **kwargs):
        raise NotImplementedError("Stream complete not implemented.")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="OpenVINOVLM", context_window=self.context_window, num_output=self.max_new_tokens)
