#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point that wraps the last Jupyter cell into a runnable script.
Relies on the following local modules:
  - multiagents_workflow.py (defines MultiAgentWorkflow)
  - gradio_helper.py       (builds the Gradio UI and wires streaming)
  - mcp_server.py, search_mcp_server.py, video_mcp_server.py (run separately)
"""

import asyncio
from multiagents_workflow import MultiAgentWorkflow
from gradio_helper import make_demo

# Instantiate the workflow once (so initialization can be one-time)
agent = MultiAgentWorkflow()

async def run_fn(chat_history, log_window, memory, cart_state, user_msg):
    """
    Streaming handler used by the Gradio UI.

    Inputs (from UI):
      - chat_history: prior chat messages (list[dict])
      - log_window:   full textual reasoning log (string)
      - memory:       reserved (unused placeholder for now)
      - cart_state:   markdown for cart panel
      - user_msg:     latest user message (string)

    Yields tuples in the exact order Gradio expects:
      (chat_history, log_window, cart_markdown, images)
    """
    # 1) one-time initialize
    if not getattr(agent, "_initialized", False):
        await agent.initialize()
        agent._initialized = True

    # 2) normalize prior state
    chat   = chat_history or []    # will feed into gr.Chatbot
    logs   = log_window   or ""    # will feed into gr.Textbox
    cart   = cart_state   or ""    # will feed into gr.Markdown
    images = []                    # will feed into gr.Gallery

    # 3) immediately show the user message
    chat.append({"role": "user", "content": user_msg or ""})
    yield chat, logs, cart, images

    # 4) stream the four-tuples from your workflow
    async for log_step, assistant_msgs, new_cart, new_images in agent.query(user_msg, memory):
        logs   += log_step + "\n"          # append to log window
        cart    = new_cart                 # update cart markdown
        images  = new_images or images     # update gallery list

        # append only the actual assistant replies
        for m in assistant_msgs:
            content = m.get("content","")
            if content and not content.startswith(("Thought:","Action:")):
                chat.append({"role":"assistant","content":content})

        # 5) yield in the exact order Gradio expects:
        #    1st → chat,  2nd → log,  3rd → cart,  4th → images
        yield chat, logs, cart, images

    # 6) final settle in case you need one last push
    yield chat, logs, cart, images


# No-op stop function (Gradio hook)
def stop_fn():
    pass


def main():
    # NOTE: Change the demo video path here if needed.
    example_video_path = "video_data/test.webm"
    demo = make_demo(example_video_path, run_fn, stop_fn)
    # Launch Gradio; share=False by default (edit as desired)
    demo.launch()


if __name__ == "__main__":
    # On Windows with Python 3.8+, ensure compatible asyncio policy when needed.
    try:
        import sys, platform
        if platform.system() == "Windows" and sys.version_info >= (3, 8):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
    main()
