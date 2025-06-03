# %% [markdown]
# ## Visual-language assistant with Phi3-Vision and OpenVINO
# 
# The Phi-3-Vision is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures. More details about model can be found in [model blog post](https://azure.microsoft.com/en-us/blog/new-models-added-to-the-phi-3-family-available-on-microsoft-azure/), [technical report](https://aka.ms/phi3-tech-report), [Phi-3-cookbook](https://github.com/microsoft/Phi-3CookBook)
# 
# In this tutorial we consider how to use Phi-3-Vision model to build multimodal chatbot using [Optimum Intel](https://github.com/huggingface/optimum-intel). And how to perform inference using [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) that provides easy-to-use API. Additionally, we optimize model to low precision using [NNCF](https://github.com/openvinotoolkit/nncf)
# #### Table of contents:
# 
# - [Prerequisites](#Prerequisites)
# - [Select Model](#Select-Model)
# - [Convert and Optimize model](#Convert-and-Optimize-model)
#     - [Compress model weights to 4-bit](#Compress-model-weights-to-4-bit)
# - [Select inference device](#Select-inference-device)
# - [Run OpenVINO model](#Run-OpenVINO-model)
# - [Interactive demo](#Interactive-demo)
# 
# 
# ### Installation Instructions
# 
# This is a self-contained example that relies solely on its own code.
# 
# We recommend  running the notebook in a virtual environment. You only need a Jupyter server to start.
# For details, please refer to [Installation Guide](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide).
# 
# <img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/phi-3-vision/phi-3-vision.ipynb" />
# 

# %% [markdown]
# ## Prerequisites
# [back to top ⬆️](#Table-of-contents:)
# 
# install required packages and setup helper functions.

# %%
import platform


%pip install -qU "git+https://github.com/huggingface/optimum-intel.git" --extra-index-url https://download.pytorch.org/whl/cpu
%pip install -q -U "torch>=2.1" "torchvision" "transformers>=4.45,<4.49" "protobuf>=3.20" "gradio>=4.26" "Pillow" "accelerate" "tqdm"  --extra-index-url https://download.pytorch.org/whl/cpu
%pip install -qU --pre --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly "openvino>=2025.0.0" "openvino-tokenizers>=2025.0.0" "openvino-genai>=2025.0.0"
%pip install -q -U "nncf>=2.15.0"


if platform.system() == "Darwin":
    %pip install -q "numpy<2.0"

# %%
import requests
from pathlib import Path

if not Path("cmd_helper.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py")
    open("cmd_helper.py", "w").write(r.text)


if not Path("gradio_helper.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/phi-3-vision/gradio_helper.py")
    open("gradio_helper.py", "w").write(r.text)

if not Path("notebook_utils.py").exists():
    r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
    open("notebook_utils.py", "w").write(r.text)

# Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
from notebook_utils import collect_telemetry

collect_telemetry("phi-3-vision.ipynb")

# %% [markdown]
# ## Select Model
# [back to top ⬆️](#Table-of-contents:)
# 
# The tutorial supports the following models from Phi-3 model family:
# - [Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
# - [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
# 
# You can select one from the provided options below.

# %%
import ipywidgets as widgets

# Select model
model_ids = [
    "microsoft/Phi-3.5-vision-instruct",
    "microsoft/Phi-3-vision-128k-instruct",
]

model_dropdown = widgets.Dropdown(
    options=model_ids,
    value=model_ids[0],
    description="Model:",
    disabled=False,
)

model_dropdown

# %%
model_id = model_dropdown.value
print(f"Selected {model_id}")
MODEL_DIR = Path(model_id.split("/")[-1])

# %% [markdown]
# ## Convert and Optimize model
# [back to top ⬆️](#Table-of-contents:)
# 
# Phi-3-vision is PyTorch model. OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate Representation (IR). [OpenVINO model conversion API](https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model) should be used for these purposes. `ov.convert_model` function accepts original PyTorch model instance and example input for tracing and returns `ov.Model` representing this model in OpenVINO framework. Converted model can be used for saving on disk using `ov.save_model` function or directly loading on device using `core.compile_model`. 
# 
# For convenience, we will use OpenVINO integration with HuggingFace Optimum. 🤗 [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) is the interface between the 🤗 Transformers and Diffusers libraries and the different tools and libraries provided by Intel to accelerate end-to-end pipelines on Intel architectures.
# 
# Among other use cases, Optimum Intel provides a simple interface to optimize your Transformers and Diffusers models, convert them to the OpenVINO Intermediate Representation (IR) format and run inference using OpenVINO Runtime. `optimum-cli` provides command line interface for model conversion and optimization. 
# 
# General command format:
# 
# ```bash
# optimum-cli export openvino --model <model_id_or_path> --task <task> <output_dir>
# ```
# 
# where task is task to export the model for, if not specified, the task will be auto-inferred based on the model. You can find a mapping between tasks and model classes in Optimum TaskManager [documentation](https://huggingface.co/docs/optimum/exporters/task_manager). Additionally, you can specify weights compression using `--weight-format` argument with one of following options: `fp32`, `fp16`, `int8` and `int4`. Fro int8 and int4 [nncf](https://github.com/openvinotoolkit/nncf) will be used for  weight compression. More details about model export provided in [Optimum Intel documentation](https://huggingface.co/docs/optimum/intel/openvino/export#export-your-model).
# 
# 
# ### Compress model weights to 4-bit
# [back to top ⬆️](#Table-of-contents:)
# For reducing memory consumption, weights compression optimization can be applied using [NNCF](https://github.com/openvinotoolkit/nncf) during run Optimum Intel CLI.
# 
# <details>
#     <summary><b>Click here for more details about weight compression</b></summary>
# Weight compression aims to reduce the memory footprint of a model. It can also lead to significant performance improvement for large memory-bound models, such as Large Language Models (LLMs). LLMs and other models, which require extensive memory to store the weights during inference, can benefit from weight compression in the following ways:
# 
# * enabling the inference of exceptionally large models that cannot be accommodated in the memory of the device;
# 
# * improving the inference performance of the models by reducing the latency of the memory access when computing the operations with weights, for example, Linear layers.
# 
# [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) provides 4-bit / 8-bit mixed weight quantization as a compression method primarily designed to optimize LLMs. The main difference between weights compression and full model quantization (post-training quantization) is that activations remain floating-point in the case of weights compression which leads to a better accuracy. Weight compression for LLMs provides a solid inference performance improvement which is on par with the performance of the full model quantization. In addition, weight compression is data-free and does not require a calibration dataset, making it easy to use.
# 
# `nncf.compress_weights` function can be used for performing weights compression. The function accepts an OpenVINO model and other compression parameters. Compared to INT8 compression, INT4 compression improves performance even more, but introduces a minor drop in prediction quality.
# 
# More details about weights compression, can be found in [OpenVINO documentation](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html).
# </details>

# %%
to_compress = widgets.Checkbox(value=True, description="Compress model", disabled=False)

to_compress

# %%
from cmd_helper import optimum_cli

model_dir = MODEL_DIR / "INT4" if to_compress.value else MODEL_DIR / "FP16"
if not model_dir.exists():
    optimum_cli(
        model_id, model_dir, additional_args={"weight-format": "int4" if to_compress.value else "fp16", "trust-remote-code": "", "task": "image-text-to-text"}
    )

# %% [markdown]
# ## Select inference device
# [back to top ⬆️](#Table-of-contents:)

# %%
from notebook_utils import device_widget

device = device_widget(default="AUTO")

device

# %% [markdown]
# ## Run OpenVINO model
# [back to top ⬆️](#Table-of-contents:)
# For inference, we will use [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) that provides easy-to-use API for running text generation. Firstly we will create pipeline with `VLMPipeline`. You can see more details in [Python vlm_chat_sample that supports VLM models](https://github.com/openvinotoolkit/openvino.genai/blob/releases/2025/0/samples/python/visual_language_chat/README.md). Also we convert the input image to ov.Tensor using `read_images` function.

# %%
import openvino_genai as ov_genai


pipe = ov_genai.VLMPipeline(model_dir, device.value)

# %%
import requests
from PIL import Image

image_path = Path("cat.png")

if not image_path.exists():
    url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
    image = Image.open(requests.get(url, stream=True).raw)
    image.save(image_path)
else:
    image = Image.open(image_path)

print("Question:\n What is unusual on this picture?")
image

# %%
import numpy as np
from openvino import Tensor


def read_image(path: str) -> Tensor:
    """

    Args:
        path: The path to the image.

    Returns: the ov.Tensor containing the image.

    """
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


rgbs = read_images(image_path)

# %%
config = ov_genai.GenerationConfig()
config.max_new_tokens = 150
config.do_sample = False
config.set_eos_token_id(pipe.get_tokenizer().get_eos_token_id())


def streamer(subword: str) -> bool:
    print(subword, end="", flush=True)


prompt = "What is unusual on this picture? Give me full explanation."
print("Answer:")

result = pipe.generate(prompt=prompt, images=rgbs, generation_config=config, streamer=streamer)

# %% [markdown]
# ## Interactive demo
# [back to top ⬆️](#Table-of-contents:)

# %%
from gradio_helper import make_demo

demo = make_demo(pipe, read_images, model_id)

try:
    demo.launch(debug=True, height=600)
except Exception:
    demo.launch(debug=True, share=True, height=600)
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/


