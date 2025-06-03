
import argparse
import io
import logging
import sys
import time
import warnings
from io import StringIO
from pathlib import Path
from typing import Tuple, Callable

import gradio as gr
import nest_asyncio
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import requests
import yaml
from llama_index.core import PromptTemplate
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
# from llama_index.core.tools import FunctionToolSpec
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
# from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.llms import MessageRole
from llama_index.core.callbacks import CallbackManager
# Agent tools
from tools import PaintCalculator, ShoppingCart
from system_prompt import react_system_header_str

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from types import SimpleNamespace



# Initialize logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

#Filter unnecessary warnings for demonstration
warnings.filterwarnings("ignore")

ov_config = {
    hints.performance_mode(): hints.PerformanceMode.LATENCY,
    streams.num(): "1",
    props.cache_dir(): ""
}

from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Existing local MCP server
local_mcp_client = BasicMCPClient("http://localhost:8000/sse")
local_mcp_spec = McpToolSpec(client=local_mcp_client)

# New DuckDuckGo MCP server
search_mcp_client = BasicMCPClient("http://localhost:8001/sse")
search_mcp_spec = McpToolSpec(client=search_mcp_client)

from openvino import Tensor
from PIL import Image
import numpy as np
import openvino_genai as ov_genai

from llama_index.core.llms import LLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.chat_engine.types import ChatMessage
from pydantic import PrivateAttr
from types import SimpleNamespace

import numpy as np
from openvino import Tensor
import numpy as np


from llama_index.core.llms import LLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms import ChatResponse, ChatMessage  # Add these imports
from llama_index.core.agent.react.base import ReActAgentWorker
from openvino import Tensor
import numpy as np
import openvino_genai as ov_genai
from pydantic import PrivateAttr
from types import SimpleNamespace

import threading
import json

from llama_index.core.agent.types import Task, TaskStep
from llama_index.core.memory import ChatMemoryBuffer

import re


def messages_to_prompt(messages):
    """Use only the latest user message to form a clean prompt."""
    for message in reversed(messages):
        if message.role == "user":
            return f"User: {message.content}\nAssistant:"
    return "User: \nAssistant:"



class VLMWrapper(LLM):

    _pipe: ov_genai.VLMPipeline = PrivateAttr()
    _config: ov_genai.GenerationConfig = PrivateAttr()

    def __init__(self, model_dir: str, device: str, system_prompt: str):
        super().__init__()
        self._pipe = ov_genai.VLMPipeline(model_dir, device)
        self._config = ov_genai.GenerationConfig(max_new_tokens=500)
        self._config.set_eos_token_id(self._pipe.get_tokenizer().get_eos_token_id())
        self._config.do_sample = False
        self._system_prompt = system_prompt
        # Dummy black image of shape (1, H, W, C) as expected
        dummy_array = np.zeros((1, 224, 224, 3), dtype=np.uint8)
        self._dummy_image = Tensor(dummy_array)
        self._lock = threading.Lock()

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=512,
            model_name="phi-3.5-vision",
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @staticmethod
    def messages_to_prompt(messages):
        """Convert a list of chat messages to a single prompt string."""
        prompt = ""
        for message in messages:
            if message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        prompt += "Assistant: "
        return prompt


    def stream_complete(self, prompt: str, **kwargs):
        output_text = ""
        with self._lock:
            result = self._pipe.generate(prompt, images=[self._dummy_image], generation_config=self._config)
            # result = self._pipe.generate(prompt, generation_config=self._config)
            print(" Raw model output:\n", result)

        for word in result.texts[0].split():
            output_text += word + " "
            yield SimpleNamespace(
                text=output_text,
                delta=word + " ",
                additional_kwargs={}
            )

    # def stream_chat(self, messages, **kwargs) -> CompletionResponseGen:
    #     prompt = messages_to_prompt(messages)
    #     print("🔍 Prompt sent to VLMPipeline:\n", prompt)
    #     for chunk in self.stream_complete(prompt, **kwargs):
    #         message = ChatMessage(role="assistant", content=chunk.text)
    #         yield ChatResponse(message=message, delta=chunk.delta)


    def _build_react_prompt(self, messages):
        conversation = ""
        for message in messages:
            if message.role == "user":
                conversation += f"User: {message.content}\n"
            elif message.role == "assistant":
                conversation += f"Assistant: {message.content}\n"
        return f"{self._system_prompt}\n\n## Current Conversation\n{conversation}Assistant:"


    def stream_chat(self, messages, tools=None, **kwargs):
        from llama_index.core.chat_engine.types import ChatMessage

        full_history = messages[:]
        observation = None
        response_text = ""

        tool_error_count = 0
        MAX_TOOL_ERRORS = 3
        
        while True:
            prompt = self._build_react_prompt(full_history)
            if observation:
                prompt += f"\nObservation: {observation}\n"

            # 🔄 Call the model
            print("📨 Prompt sent to model:\n", prompt)
            with self._lock:
                result = self._pipe.generate(prompt, images=[self._dummy_image], generation_config=self._config)
                # result = self._pipe.generate(prompt, generation_config=self._config)
                print(" Raw model output:\n", result)

            output = result.texts[0].strip()
            response_text += output

            # 🔎 Yield streaming output
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=response_text),
                delta=output
            )

            # Check for final answer (no more tools needed)
            if "Answer:" in output:
                break

            try:
                # Extract "Action:" and "Action Input:" lines from the model output
                action_line = next((line for line in output.splitlines() if line.startswith("Action:")), None)
                input_line = next((line for line in output.splitlines() if line.startswith("Action Input:")), None)

                if action_line:
                    action = action_line.split("Action:", 1)[1].strip()
                else:
                    log.warning("No 'Action:' line found in model response.")
                    action = None  # or fallback

                if input_line:
                    action_input_str = input_line.split("Action Input:", 1)[1].strip()
                else:
                    log.warning("No 'Action Input:' line found in model response.")
                    action_input_str = None  # or fallback

                available_tool_names = []

                if tools is not None:
                    available_tool_names = [t.metadata.name for t in tools]
                    print(f"Available tools: {available_tool_names}")
                    print(f"Invoking tool: '{action}' with raw input: {action_input_str}")
                else:
                    log.warning("No tools were registered or passed to the agent.")

                if action is None or action_input_str is None or action not in available_tool_names:
                    observation = "Tool format was invalid or the model skipped tool usage."
                    full_history.append(ChatMessage(role="user", content=f"Observation: {observation}"))
                    continue  # Skip tool invocation and go to next model call

                # Parse action input (expecting JSON string)
                try:
                    action_input = json.loads(action_input_str)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON input: {action_input_str}")

                tool = next(t for t in tools if t.metadata.name == action)
                tool_result = tool(**action_input)
                observation = tool_result if isinstance(tool_result, str) else str(tool_result)
                print(f"Tool result: {observation}")

                full_history.append(ChatMessage(role="user", content=f"Observation: {observation}"))

            except Exception as e:
                tool_error_count += 1
                import traceback
                print(f" Tool execution error: {type(e).__name__}: {e}")
                traceback.print_exc()
                observation = f"Tool error: {e}"
                full_history.append(ChatMessage(role="user", content=f"Observation: {observation}"))
                if tool_error_count >= MAX_TOOL_ERRORS:
                    break



    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        text = self._pipe.generate(prompt, images=[self._dummy_image], generation_config=self._config).texts[0]
        return CompletionResponse(text=text)

    def chat(self, messages, **kwargs):
        raise NotImplementedError("chat not supported")

    def astream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("astream_complete not implemented")

    def acomplete(self, prompt: str, **kwargs):
        raise NotImplementedError("acomplete not implemented")

    def astream_chat(self, messages, **kwargs):
        raise NotImplementedError("astream_chat not implemented")

    def achat(self, messages, **kwargs):
        raise NotImplementedError("achat not implemented")
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow the agent to call this like a regular LLM."""
        return self.complete(prompt, **kwargs).text
    
    

def setup_models(chat_model: Path, embedding_model: Path, device: str):
    embedding = OpenVINOEmbedding(str(embedding_model), device=device)
    log.info(" Initialized OpenVINOEmbedding")
    return embedding



def setup_tools() -> Tuple:
    """
    Fetch tools from both local and search MCP servers.
    """
    # tools = local_mcp_spec.to_tool_list() + search_mcp_spec.to_tool_list()
    tools = local_mcp_spec.to_tool_list()
    return tuple(tools)


def load_documents(text_example_en_path: Path) -> VectorStoreIndex:
    """
    Loads documents from the given path
    
    Args:
        text_example_en_path: Path to the document to load
        
    Returns:
        VectorStoreIndex for the loaded documents
    """
    
    if not text_example_en_path.exists():
        text_example_en = "test_painting_llm_rag.pdf"
        r = requests.get(text_example_en)
        content = io.BytesIO(r.content)
        with open(text_example_en_path, "wb") as f:
            f.write(content.read())

    reader = SimpleDirectoryReader(input_files=[text_example_en_path])
    print("🔹 About to call reader.load_data()")
    documents = reader.load_data()
    print(" Finished reader.load_data()")

    print("First document preview:", documents[0].text[:200])  # Truncate for readability
    for doc in documents:
        print("Document snippet:", doc.text[:10])

    print("🔹 About to build VectorStoreIndex")
    index = VectorStoreIndex.from_documents(documents)
    print(" Finished building VectorStoreIndex")

    return index

def custom_handle_reasoning_failure(callback_manager: CallbackManager, exception: Exception):
    """
    Provides custom error handling for agent reasoning failures.
    
    Args:
        callback_manager: The callback manager instance for event handling
        exception: The exception that was raised during reasoning
    """
    return "Hmm...I didn't quite that. Could you please rephrase your question to be simpler?"


def run_app(agent: ReActAgent, public_interface: bool = False) -> None:
    """
    Launches the application with the specified agent and interface settings.
    
    Args:
        agent: The ReActAgent instance configured with tools
        public_interface: Whether to launch with a public-facing Gradio interface
    """
    class Capturing(list):
        """A context manager that captures stdout output into a list."""
        def __enter__(self):
            """
            Redirects stdout to a StringIO buffer and returns self.
            Called when entering the 'with' block.
            """
            self._stdout = sys.stdout
            sys.stdout = self._stringio = StringIO()
            return self
        def __exit__(self, *args):
            """
            Stores captured output in this list and restores stdout.
            Called when exiting the 'with' block.
            """
            self.extend(self._stringio.getvalue().splitlines())
            del self._stringio
            sys.stdout = self._stdout        

    def _handle_user_message(user_message, history):
        return "", [*history, (user_message, None)]

    def update_cart_display()-> str:
        """
        Generates an HTML representation of the shopping cart contents.
        
        Retrieves current cart items and creates a formatted HTML table
        showing product details, quantities, prices, and totals.
        If the cart is empty, returns a message indicating this.
        
        Returns:
            str: Markdown-formatted HTML table of cart contents
                or message indicating empty cart
        """
        cart_items = ShoppingCart.get_cart_items()
        if not cart_items:
            return "### 🛒 Your Shopping Cart is Empty"
            
        table = "### 🛒 Your Shopping Cart\n\n"
        table += "<table>\n"
        table += "  <thead>\n"
        table += "    <tr>\n"
        table += "      <th>Product</th>\n"
        table += "      <th>Qty</th>\n"
        table += "      <th>Price</th>\n"
        table += "      <th>Total</th>\n"
        table += "    </tr>\n"
        table += "  </thead>\n"
        table += "  <tbody>\n"
            
        for item in cart_items:
            table += "    <tr>\n"
            table += f"      <td>{item['product_name']}</td>\n"
            table += f"      <td>{item['quantity']}</td>\n"
            table += f"      <td>${item['price_per_unit']:.2f}</td>\n"
            table += f"      <td>${item['total_price']:.2f}</td>\n"
            table += "    </tr>\n"
            
        table += "  </tbody>\n"
        table += "</table>\n"
        
        total = sum(item["total_price"] for item in cart_items)
        table += f"\n**Total: ${total:.2f}**"
        return table

    def _generate_response(chat_history: list, log_history: list | None = None)->Tuple[str,str,str]:
        """
        Generate a streaming response from the agent with formatted thought process logs.
        
        This function:
        1. Captures the agent's thought process
        2. Formats the thought process into readable logs
        3. Streams the agent's response token by token
        4. Tracks performance metrics for thought process and response generation
        5. Updates the shopping cart display
        
        Args:
            chat_history: List of conversation messages
            log_history: List to store logs, will be initialized if None
            
        Yields:
            tuple: (chat_history, formatted_log_history, cart_content)
                - chat_history: Updated with agent's response
                - formatted_log_history: String of joined logs
                - cart_content: HTML representation of the shopping cart
        """
        log.info(f"log_history {log_history}")           

        if not isinstance(log_history, list):
            log_history = []

        # Capture time for thought process
        start_thought_time = time.time()

        query = chat_history[-1][0]

        # Create a temporary memory buffer to satisfy the required field
        memory = ChatMemoryBuffer.from_defaults()

        task = Task(input=query, memory=memory)
        step = agent.initialize_step(task)
        step_output = agent.stream_step(step=step, task=task)

        # Extract response string
        response = step_output.output.response

        if "Action:" not in response or "Action Input:" not in response:
            log.warning("Model response skipped tool call. Showing raw output.")
            formatted_output = ["\n**Model Response:**\n" + str(response)]
            return formatted_output, 0  # skip tool parsing

        # Otherwise, go on parsing
        output = response.split("\n")
        formatted_output = []

        print("Thought process log:")

        for line in output:
            print(" ", line)
            if "Thought:" in line:
                formatted_output.append("\n **Thought:**\n" + line.split("Thought:", 1)[1])
            elif "Action:" in line:
                formatted_output.append("\n **Action:**\n" + line.split("Action:", 1)[1])
            elif "Action Input:" in line:
                formatted_output.append("\n **Input:**\n" + line.split("Action Input:", 1)[1])
            elif "Observation:" in line:
                formatted_output.append("\n **Result:**\n" + line.split("Observation:", 1)[1])
            else:
                formatted_output.append(line)

        end_thought_time = time.time()
        thought_process_time = end_thought_time - start_thought_time

        # After response is complete, show the captured logs in the log area
        log_entries = "\n".join(formatted_output)
        log_history.append("### Agent's Thought Process")
        thought_process_log = f"Thought Process Time: {thought_process_time:.2f} seconds"
        log_history.append(f"{log_entries}\n{thought_process_log}")
        cart_content = update_cart_display() # update shopping cart
        yield chat_history, "\n".join(log_history), cart_content  # Yield after the thought process time is captured

        # Now capture response generation time
        start_response_time = time.time()

        chat_history[-1][1] = ""
        response_text = ""

        for chunk in response:
            word = chunk.delta if hasattr(chunk, "delta") else str(chunk)
            response_text += word
            chat_history[-1][1] = response_text
            yield chat_history, "\n".join(log_history), cart_content

        end_response_time = time.time()
        response_time = end_response_time - start_response_time

        # Log tokens per second along with the device information
        tokens = len(chat_history[-1][1].split(" ")) * 4 / 3  # Convert words to approx token count
        if response_time < 1e-3:  # protect against very small/zero division
            response_log = f"Response Time: {response_time:.2f} seconds (speed unavailable)"
        else:
            response_log = f"Response Time: {response_time:.2f} seconds ({tokens / response_time:.2f} tokens/s)"


        log.info(response_log)

        # Append the response time to log history
        log_history.append(response_log)
        yield chat_history, "\n".join(log_history), cart_content  # Join logs into a string for display

    def _reset_chat()-> tuple[str, list, str, str]:
        """
        Resets the chat interface and agent state to initial conditions.
        
        This function:
        1. Resets the agent's internal state
        2. Clears all items from the shopping cart
        3. Returns values needed to reset the UI components
        
        Returns:
            tuple: Values to reset UI components
                - Empty string: Clears the message input
                - Empty list: Resets chat history
                - Default log heading: Sets initial log area text
                - Empty cart display: Shows empty shopping cart
        """
        agent.reset()
        ShoppingCart._cart_items = []
        return "", [], " Agent's Thought Process", update_cart_display()

    def run()-> None:
        """
        Sets up and launches the Gradio web interface for the Smart Retail Assistant.
        
        This function:
        1. Loads custom CSS styling if available
        2. Configures the Gradio theme and UI components
        3. Sets up the chat interface with agent interaction
        4. Configures event handlers for user inputs
        5. Adds example prompts for users
        6. Launches the web interface
        
        The interface includes:
        - Chat window for user-agent conversation
        - Log window to display agent's thought process
        - Shopping cart display
        - Text input for user messages
        - Submit and Clear buttons
        - Sample questions for easy access
        """
        custom_css = ""
        try:
            with open("css/gradio.css", "r") as css_file:
                custom_css = css_file.read()            
        except Exception as e:            
            log.warning(f"Could not load CSS file: {e}")

        theme = gr.themes.Default(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Montserrat"), "ui-sans-serif", "sans-serif"],
        )

        print("🔹 Before gr.Blocks")

        with gr.Blocks(theme=theme, css=custom_css) as demo:
            print(" Entered gr.Blocks")

            header = gr.HTML(
                        "<div class='intel-header-wrapper'>"
                        "  <div class='intel-header'>"
                        "    <img src='https://www.intel.com/content/dam/logos/intel-header-logo.svg' class='intel-logo'></img>"
                        "    <div class='intel-title'>Smart Retail Assistant 🤖: Agentic LLMs with RAG 💭</div>"
                        "  </div>"
                        "</div>"
            )
            print(" Created header")

            with gr.Row():
                chat_window = gr.Chatbot(
                    label="Paint Purchase Helper",
                    avatar_images=(None, "https://docs.openvino.ai/2024/_static/favicon.ico"),
                    height=400,  # Adjust height as per your preference
                    scale=2  # Set a higher scale value for Chatbot to make it wider
                    #autoscroll=True,  # Enable auto-scrolling for better UX
                )            
                log_window = gr.Markdown(                                                                    
                        show_label=True,                        
                        value="###  Agent's Thought Process",
                        height=400,                        
                        elem_id="agent-steps"
                )
                cart_display = gr.Markdown(
                    value=update_cart_display(),
                    elem_id="shopping-cart",
                    height=400
                )
                print(" Created chat, log, cart")

            with gr.Row():
                message = gr.Textbox(label="Ask the Paint Expert", scale=4, placeholder="Type your prompt/Question and press Enter")

                with gr.Column(scale=1):
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear = gr.ClearButton()
                          
            sample_questions = [
                "what paint is the best for kitchens?",
                "what is the price of it?",
                "how many gallons of paint do I need to cover 600 sq ft ?",
                "add them to my cart",
                "what else do I need to complete my project?",
                "add 2 brushes to my cart",
                "create a table with paint products sorted by price",
                "Show me what's in my cart",
                "clear shopping cart",
                "I have a room 1000 sqft, I'm looking for supplies to paint the room"              
            ]
            gr.Examples(
                examples=sample_questions,
                inputs=message, 
                label="Examples"
            )                     
            
            # Ensure that individual components are passed
            message.submit(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False                
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],
                outputs=[chat_window, log_window, cart_display],
            )

            submit_btn.click(
                _handle_user_message,
                inputs=[message, chat_window],
                outputs=[message, chat_window],
                queue=False,
            ).then(
                _generate_response,
                inputs=[chat_window, log_window],
                outputs=[chat_window, log_window, cart_display],
            )
            clear.click(_reset_chat, None, [message, chat_window, log_window, cart_display])

            gr.Markdown("------------------------------")            

        log.info("Demo is ready!")
        print(" Gradio is about to launch. If the formatted system prompt didn't print above, check PromptTemplate rendering.")
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=public_interface,
            debug=True,
            inbrowser=True  # <-- Force browser to open
        )

    run()


from types import SimpleNamespace

def run(chat_model: Path, embedding_model: Path, rag_pdf: Path, device: str, public_interface: bool = False):
    """
    Initializes and runs the agentic rag solution
    """
    # Load models and embedding
    embedding = setup_models(chat_model, embedding_model, device)
    Settings.embed_model = embedding

    # RAG setup
    text_example_en_path = Path(rag_pdf)
    print("[INFO] About to build VectorStoreIndex")
    index = load_documents(text_example_en_path)
    log.info(f"loading in {index}")

    nest_asyncio.apply()

    # Escape braces to avoid format errors in tool descriptions
    def escape_braces(text: str) -> str:
        return text.replace("{", "{{{{").replace("}", "}}}}")

    # Phase 1: Define only the metadata for vector_tool
    vector_tool_metadata = ToolMetadata(
        name="paint_qa",
        description="""
        Use this tool for ANY question related to **paint**, **painting**, or **supplies**.

        WHEN TO USE:
        - Questions about kitchen paint
        - Questions about paint brands or finishes
        - Coverage area or gallons needed
        - What products to buy
        - What supplies are required

        EXAMPLES:
        - What paint is best for kitchens?
        - How many gallons to paint 500 sq ft?
        - What finish is recommended for bathrooms?
        - What brushes do I need?

        DO NOT try to answer these yourself — always call this tool first.
        """
    )

    # Create mock tool for prompt rendering
    actual_tools = list(setup_tools())
    temp_tool = SimpleNamespace(metadata=vector_tool_metadata)
    tool_list = [temp_tool] + actual_tools

    # Build tool description for system prompt
    escaped_descriptions = []
    for tool in tool_list:
        try:
            escaped = escape_braces(tool.metadata.description)
            print(f"[INFO] Escaped {tool.metadata.name} description:\n{escaped}\n")
            escaped_descriptions.append(f"{tool.metadata.name}: {escaped}")
        except Exception as e:
            print(f"[WARN] Error escaping {tool.metadata.name}: {e}")

    tool_desc_str = "\n".join(escaped_descriptions)
    tool_names = [tool.metadata.name for tool in tool_list]
    log.info(f"Registered tools: {tool_names}")

    # Render system prompt
    rendered_prompt = react_system_header_str.replace("{tool_desc}", tool_desc_str)
    print("[INFO] Final system prompt:\n", rendered_prompt)

    # Init VLM with rendered prompt
    llm = VLMWrapper(str(chat_model), device, system_prompt=rendered_prompt)
    Settings.llm = llm

    # Phase 2: Now we can safely call index.as_query_engine()
    vector_tool = QueryEngineTool(
        index.as_query_engine(streaming=True),
        metadata=vector_tool_metadata
    )

    tools = (vector_tool,) + tuple(actual_tools)

    # Patch to detect repeated tool calls (loop guard)
    def custom_handle_reasoning_failure(state):
        if not hasattr(state, "loop_guard"):
            state.loop_guard = set()

        action = getattr(state, "last_action", None)
        if action:
            sig = (action.tool, json.dumps(action.tool_input, sort_keys=True))
            if sig in state.loop_guard:
                log.warning("[WARN] Detected repeated action. Breaking loop.")
                return "Thought: Repeated tool call detected.\nAnswer: Sorry, I'm unable to proceed due to repeated reasoning failures."
            state.loop_guard.add(sig)

        return None

    # Regex-based normalizer to fix bad model capitalization and spacing
    class CaseNormalizerParser:
        def parse(self, output: str):
            import re
            substitutions = {
                r"(?i)^ *action *:": "Action:",
                r"(?i)^ *action_input *:": "Action Input:",
                r"(?i)^ *thought *:": "Thought:",
                r"(?i)^ *answer *:": "Answer:",
                r"(?i)^ *observation *:": "Observation:",
            }
            lines = output.splitlines()
            fixed_lines = []
            for line in lines:
                fixed = line
                for pattern, replacement in substitutions.items():
                    if re.match(pattern, line.strip()):
                        fixed = re.sub(pattern, replacement, line.strip(), count=1)
                        break
                fixed_lines.append(fixed)
            fixed_output = "\n".join(fixed_lines)

            print("[DEBUG] Before normalization:\n", output)
            print("[DEBUG] After normalization:\n", fixed_output)

            from llama_index.core.agent.react.output_parser import ReActOutputParser
            return ReActOutputParser().parse(fixed_output)

    # Build ReAct agent
    agent = ReActAgentWorker.from_tools(
        tools=tools,
        llm=Settings.llm,
        max_iterations=5,
        handle_reasoning_failure_fn=custom_handle_reasoning_failure,
        react_chat_formatter=ReActChatFormatter.from_defaults(
            observation_role=MessageRole.TOOL
        ),
        verbose=True,
        system_prompt=PromptTemplate(rendered_prompt)
    )

    def normalized_step(self, state, input_str):
        raw_output = self._call_llm(state, input_str)
        print("[DEBUG] Raw model output:\n", raw_output)

        try:
            parsed_step = CaseNormalizerParser().parse(raw_output)

            # This tells the agent a tool was called
            if parsed_step.action:
                state.last_action = parsed_step.action

            return parsed_step
        except Exception as e:
            print("[ERROR] CaseNormalizerParser failed:", e)
            raise e


    # Patch directly
    agent._step = normalized_step.__get__(agent)

    # Launch the Gradio app
    run_app(agent, public_interface)


if __name__ == "__main__":
    # Define the argument parser at the end
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="model/qwen2-7B-INT4", help="Path to the chat model directory")
    parser.add_argument("--embedding_model", type=str, default="model/bge-large-FP32", help="Path to the embedding model directory")
    parser.add_argument("--rag_pdf", type=str, default="data/test_painting_llm_rag.pdf", help="Path to a RAG PDF file with additional knowledge the chatbot can rely on.")    
    parser.add_argument("--device", type=str, default="AUTO:GPU,CPU", help="Device for inferencing (CPU,GPU,GPU.1,NPU)")
    parser.add_argument("--public", default=False, action="store_true", help="Whether interface should be available publicly")

    args = parser.parse_args()

    run(Path(args.chat_model), Path(args.embedding_model), Path(args.rag_pdf), args.device, args.public)
