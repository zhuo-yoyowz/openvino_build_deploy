from pathlib import Path
from PIL import Image
import numpy as np
from openvino import Tensor
import openvino_genai as ov_genai

# Path to OpenVINO model directory
model_dir = Path("model/ov_phi35_vision")
device = "AUTO"  # or "GPU" if you want to force it

# Load pipeline
pipe = ov_genai.VLMPipeline(model_dir, device)

# Load and preprocess image
def read_image(path: str) -> Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return Tensor(image_data)

image_path = "cat.png"
image_tensor = read_image(image_path)

# Configure generation
config = ov_genai.GenerationConfig(
    max_new_tokens=100,
    do_sample=False,
    eos_token_id=pipe.get_tokenizer().get_eos_token_id()
)

# Run prompt
prompt = "What is unusual on this picture? Give me a full explanation."
print("Prompt:", prompt)
print("Answer:")

# Streamer for printing token-by-token
def streamer(text: str):
    print(text, end="", flush=True)

# Run generation
pipe.generate(prompt=prompt, images=[image_tensor], generation_config=config, streamer=streamer)
