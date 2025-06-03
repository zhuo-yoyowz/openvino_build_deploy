from openvino import Tensor
import numpy as np
from pathlib import Path
from PIL import Image
import openvino_genai as ov_genai

# Model path
MODEL_PATH = Path("model/ov_phi35_vision")

# Image path
IMAGE_PATH = Path("cat.png")

# Load image
def read_image(path: Path) -> Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return Tensor(image_data)

rgbs = [read_image(IMAGE_PATH)]

# Initialize VLM pipeline
pipe = ov_genai.VLMPipeline(MODEL_PATH, device="AUTO")

# Define streaming output callback
def streamer(subword: str) -> bool:
    print(subword, end="", flush=True)

# Generation configuration
config = ov_genai.GenerationConfig()
config.max_new_tokens = 250
config.do_sample = False
config.set_eos_token_id(pipe.get_tokenizer().get_eos_token_id())

# ReAct-style prompt
prompt = """
You are a helpful assistant that uses tools to answer questions.
Use the following format:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool>

User: What is unusual on this picture?
Assistant:
"""

print("Prompt:", prompt.strip())
print("Answer:")
pipe.generate(prompt=prompt, images=rgbs, generation_config=config, streamer=streamer)
