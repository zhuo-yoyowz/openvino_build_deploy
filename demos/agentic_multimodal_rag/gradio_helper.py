from typing import Callable
import gradio as gr

examples = [
    ["What dessert is included in this video?"],
    ["Tell me how to make a trifle. I want to make one myself"],
    ["I think I need to buy some ingredients first. I need product information on online shopping platforms."],
    ["Search for custard for me in these online shopping platforms"],
]


def clear_files():
    return "Vector Store is Not ready"


def handle_user_message(message, history):
    """
    callback function for updating user messages in interface on submit button click

    Params:
      message: current message
      history: conversation history
    Returns:
      None
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]



def make_demo(
    example_path: str,
    build_index: Callable,
    search: Callable,
    run_fn: Callable,
    stop_fn: Callable,
):
    with gr.Blocks(theme=gr.themes.Soft(), css=".disclaimer{font-variant-caps:all-small-caps;}") as demo:
        # Title
        gr.Markdown("<h1><center>Smart Retail Assistant 🤖</center></h1>")
        gr.Markdown("<center>Powered by OpenVINO + MCP Tools</center>")

        # Hidden state
        image_list = gr.State([])
        txt_list   = gr.State([])

        with gr.Row():
            # === Left Column: Video + Log/Cart Below ===
            with gr.Column(scale=2):
                video_file = gr.Video(value=example_path, label="1) Upload or choose a video", interactive=True)
                build_btn  = gr.Button("2) Build Vector Store", variant="primary")
                status     = gr.Textbox("Vector store not built", interactive=False, show_label=False)

                with gr.Row():
                    #log_md  = gr.Markdown("### Agent’s Reasoning Log", label="Logs", height=300)
                    log_window = gr.Markdown("### 🤖 Agent’s Reasoning Log", label="Logs", height=300)
                    cart_md = gr.Markdown("### 🛒 Your Actions / Cart", label="Cart", height=300)

            # === Right Column: Chat UI ===
            with gr.Column(scale=2):
                chatbot   = gr.Chatbot(label="Conversation", height=500)
                with gr.Row():
                    msg      = gr.Textbox(placeholder="Type your message…", show_label=False, container=False)
                    send_btn = gr.Button("Send", variant="primary")
                    stop_btn = gr.Button("Stop")
                    clr_btn  = gr.Button("Clear")
                gr.Examples(examples, inputs=[msg], label="Click example, then Send")

        # === Events ===

        build_btn.click(fn=build_index, inputs=[video_file], outputs=[status], queue=True)

        def _chain_event(trigger):
            return (
                trigger(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False)
                       .then(search, [chatbot], [image_list, txt_list], queue=True)
                       .then(run_fn,  [chatbot, log_window, image_list, txt_list], [chatbot, log_window, cart_md], queue=True)
            )

        submit_event = _chain_event(msg.submit)
        click_event  = _chain_event(send_btn.click)

        stop_btn.click(fn=stop_fn, inputs=None, outputs=None, cancels=[submit_event, click_event], queue=False)

        clr_btn.click(
            fn=lambda: ("", [], "### Agent’s Reasoning Log", "### 🛒 Your Actions / Cart"),
            inputs=None,
            outputs=[msg, chatbot, log_window, cart_md],
            queue=False
        )

    return demo
