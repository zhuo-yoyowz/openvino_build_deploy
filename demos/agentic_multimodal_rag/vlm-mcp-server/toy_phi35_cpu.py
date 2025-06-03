from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "microsoft/Phi-3.5-vision-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="eager",  # 👈 this disables FlashAttention
).eval().to("cuda" if torch.cuda.is_available() else "cpu")

prompt = """
You are a helpful assistant that uses tools to answer questions.
Use the following format:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool>

User: what paint is the best for kitchens?
Assistant:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=300)

print(" Output:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
