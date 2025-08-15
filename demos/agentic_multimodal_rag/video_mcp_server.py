# video_mcp_server.py
from __future__ import annotations

import json
import base64
import tempfile
import uuid
from pathlib import Path

from fastmcp import FastMCP

from llama_index.core import Settings, StorageContext, SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface_openvino import OpenVINOClipEmbedding
from llama_index.multi_modal_llms.openvino import OpenVINOMultiModal
from llama_index.core.schema import ImageNode  # noqa: F401 (kept for clarity)
import qdrant_client

from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

# ── Persistent output dir for images we want the UI to load reliably ──────────
OUTPUT_DIR = Path(__file__).parent / "retrieved_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

vlm_int4_model_path = "./Phi-3.5-vision-instruct-int4-ov"

processor = AutoProcessor.from_pretrained(vlm_int4_model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(vlm_int4_model_path)

def messages_to_prompt(messages, image_documents):
    """
    Prepares the input messages and images.
    """
    images = []
    placeholder = ""

    for i, img_doc in enumerate(image_documents, start=1):
        images.append(Image.open(img_doc.image_path))
        placeholder += f"<|image_{i}|>\n"

    conversation = [
        {"role": "user", "content": placeholder + messages[0].content},
    ]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, images, return_tensors="pt")
    return inputs

# Embeddings / index setup
clip_embed = OpenVINOClipEmbedding(
    model_id_or_path="./CLIP-ViT-B-32-laion2B-s34B-b79K",
    device="GPU",
    trust_remote_code=True,
)
Settings.embed_model = clip_embed

vlm = OpenVINOMultiModal(
    model_id_or_path=vlm_int4_model_path,
    device="GPU",
    messages_to_prompt=messages_to_prompt,
    trust_remote_code=True,
    generate_kwargs={"do_sample": False, "eos_token_id": processor.tokenizer.eos_token_id},
)

client = qdrant_client.QdrantClient(":memory:")
text_store = QdrantVectorStore(client=client, collection_name="text_collection")
image_store = QdrantVectorStore(client=client, collection_name="image_collection")
storage_ctx = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

docs = SimpleDirectoryReader("./mixed_data").load_data()
index = MultiModalVectorStoreIndex.from_documents(
    docs,
    storage_context=storage_ctx,
    image_embed_model=Settings.embed_model,
    transformations=[SentenceSplitter(chunk_size=300, chunk_overlap=30)],
)
retriever = index.as_retriever(similarity_top_k=2, image_similarity_top_k=5)

# ---------- MCP App ----------
mcp = FastMCP("VideoSearchMCP")

@mcp.tool()
def search_from_video(query: str) -> str:
    # 1) Retrieve nodes
    nodes = retriever.retrieve(query)

    # 2) Collect up to 5 existing image paths from retrieved nodes
    frames = [
        str(n.node.image_path)
        for n in nodes
        if hasattr(n.node, "image_path")
        and n.node.image_path
        and Path(n.node.image_path).exists()
    ][:5]

    # 3) Resize to VLM-friendly size into temp files
    resized_tmp_paths: list[str] = []
    for fp in frames:
        with Image.open(fp) as im:
            im = im.convert("RGB").resize((448, 448))
            out = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.jpg"
            im.save(out, format="JPEG")
            resized_tmp_paths.append(str(out))

    # 4) Prepare image docs for the VLM
    image_docs = SimpleDirectoryReader(input_files=resized_tmp_paths).load_data()

    # 5) Build a concise context for the prompt
    context_text = " ".join([str(n.text or "") for n in nodes])[:500]

    # 6) Non-streaming call (returns object with .text)
    answer = vlm.complete(
        prompt=f"Context: {context_text}\nQuestion: {query}",
        image_documents=image_docs,
    ).text

    # 7) Create base64 data URLs (UI can always render these)
    data_urls: list[str] = []
    for p in resized_tmp_paths:
        try:
            b = Path(p).read_bytes()
            b64 = base64.b64encode(b).decode("utf-8")
            data_urls.append(f"data:image/jpeg;base64,{b64}")
        except Exception:
            # Ignore read errors; client still has stable paths as fallback
            pass

    # 8) Also write persistent copies (stable paths under OUTPUT_DIR)
    stable_paths: list[str] = []
    for p in resized_tmp_paths:
        try:
            src = Path(p)
            dest = OUTPUT_DIR / f"{uuid.uuid4().hex}.jpg"
            # copy bytes (don’t just rename across filesystems)
            dest.write_bytes(src.read_bytes())
            stable_paths.append(str(dest))
        except Exception:
            # If copy fails, skip; we still have data_urls
            pass

    # 9) Cleanup the temp files AFTER we created b64 + stable paths
    for p in resized_tmp_paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    # 10) Return JSON as string (MCP requires string return)
    # Provide both keys so the client can prefer base64 but still use file paths if desired.
    return json.dumps(
        {
            "answer": answer,
            "image_paths": stable_paths,  # persistent paths (prefer these over temp paths)
            "data_urls": data_urls,       # base64 data URLs (preferred in UI)
            "image_b64": data_urls,       # alias for compatibility with earlier client code
        },
        ensure_ascii=False,
    )

app = mcp.sse_app()
