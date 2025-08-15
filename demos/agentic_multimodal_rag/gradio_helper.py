from typing import Callable, Tuple, List
import gradio as gr
from pathlib import Path
import inspect
import re

examples = [
    "What dessert is included in this video?",
    "Tell me how to make a trifle. I want to make one myself",
    "I need to buy some ingredients—search online shopping platforms.",
    "Search for custard on the shopping platforms",
]

def clear_files():
    return "Vector Store is Not Ready"

def make_demo(example_path: str, 
              run_fn: Callable[[List[dict], str, list, str, str], Tuple[List[dict], str, str, List[str]]], 
              stop_fn: Callable[[], None]) -> gr.Blocks:

    example_path = Path(example_path)

    with gr.Blocks(theme=gr.themes.Soft(), css="""
        #agent-steps { border: 2px solid #ddd; border-radius: 8px; padding: 12px; }
        #shopping-cart { border: 2px solid #4CAF50; border-radius: 8px; padding: 12px; }
    """) as demo:

        gr.Markdown("<h1 style='text-align:center'>Smart Retail Assistant 🤖</h1>")
        gr.Markdown("<p style='text-align:center'>Powered by OpenVINO + MCP Tools</p>")

        with gr.Row():
            with gr.Column(scale=3):
                video = gr.Video(value=str(example_path), label="Upload / Choose video", interactive=True)
                chatbot = gr.Chatbot(type="messages", label="Conversation")
                gallery = gr.Gallery(label="Retrieved Images", type="pil")
                msg = gr.Textbox(placeholder="Type your message…", show_label=False)
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    stop_btn = gr.Button("Stop")
                    clear_btn = gr.Button("Clear All")
                gr.Examples(
                    examples=[
                        "What dessert is in the video?",
                        "Tell me how to make a trifle. I want to make one myself",
                        "I need to buy some ingredients—search online shopping platforms.",
                        "Search for custard on the shopping platforms",
                        "Show my cart"
                    ],
                    inputs=msg
                )

            with gr.Column(scale=2):
                gr.Markdown("### 🧠 Agent Reasoning Log")
                log_window = gr.Textbox(value="", label="", elem_id="agent-steps", lines=15)
                gr.Markdown("### 🛒 Actions / Cart")
                cart_md = gr.Markdown(value="", elem_id="shopping-cart", height=200)

        # Hidden state so we don't feed the textbox back into the backend
        log_state = gr.State("")

        def _normalize_for_display(s: str) -> str:
            if not isinstance(s, str):
                return ""
            s = s.replace("\r", "")
            # keep paragraph breaks, but collapse intra-paragraph newlines
            s = re.sub(r'(?<!\n)\n(?!\n)', ' ', s)
            s = re.sub(r'[ \t]+', ' ', s)
            s = re.sub(r'[ \t]+\n', '\n', s)
            return s

        async def run_passthrough(chat, _log_state, st1, st2, message):
            """
            The backend yields (chat, full_log, cart_md, images) repeatedly.
            We REPLACE the textbox with full_log every tick (no accumulation here).
            """
            result = run_fn(chat, _log_state, st1, st2, message)

            if inspect.isasyncgen(result):
                async for step in result:
                    chat_o, log_full, cart_md_o, images_o = step
                    yield chat_o, _normalize_for_display(log_full or ""), cart_md_o, images_o
                return

            if inspect.iscoroutine(result):
                chat_o, log_full, cart_md_o, images_o = await result
                yield chat_o, _normalize_for_display(log_full or ""), cart_md_o, images_o
                return

            if inspect.isgenerator(result):
                for step in result:
                    chat_o, log_full, cart_md_o, images_o = step
                    yield chat_o, _normalize_for_display(log_full or ""), cart_md_o, images_o
                return

            chat_o, log_full, cart_md_o, images_o = result
            yield chat_o, _normalize_for_display(log_full or ""), cart_md_o, images_o
            return

        send_btn.click(
            fn=run_passthrough,
            inputs=[chatbot, log_state, gr.State([]), gr.State([]), msg],
            outputs=[chatbot, log_window, cart_md, gallery],
            queue=True
        )

        stop_btn.click(fn=stop_fn)
        clear_btn.click(
            fn=lambda: ([], "", "", []),
            inputs=[],
            outputs=[chatbot, log_window, cart_md, gallery]
        )

        # IMPORTANT: enable queue globally so Gradio streams progressively
        demo.queue()

    return demo
