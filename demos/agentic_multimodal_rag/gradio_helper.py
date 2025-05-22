from typing import Callable
import gradio as gr

examples = [
    ["Tell me more about gaussian function"],
    ["Explain the formula of gaussian function to me"],
    ["What is the Herschel Maxwell derivation of a Gaussian ?"],
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
    """
    example_path: Path or URL to a sample video
    build_index:   fn(video_path) -> Text that updates status when vector store is ready
    search:        fn(chat_history) -> (image_list, txt_list)
    run_fn:        fn(chat_history, log_history) -> (chat_history, log_history, cart_markdown)
    stop_fn:       fn() cancels any in-flight generation
    """
    with gr.Blocks(theme=gr.themes.Soft(), css=".disclaimer{font-variant-caps:all-small-caps;}") as demo:
        # Title
        gr.Markdown("<h1><center>Smart Retail Assistant 🤖</center></h1>")
        gr.Markdown("<center>Powered by OpenVINO + MCP Tools</center>")

        # Hidden state for your multimodal RAG
        image_list = gr.State([])
        txt_list   = gr.State([])

        # === Three-column layout ===
        with gr.Row():
            # — Left: Video Uploader + Chat UI —
            with gr.Column(scale=2):
                video_file   = gr.Video(value=example_path, label="1) Upload or choose a video", interactive=True)
                build_btn    = gr.Button("2) Build Vector Store", variant="primary")
                status       = gr.Textbox("Vector store not built", interactive=False, show_label=False)

                chatbot      = gr.Chatbot(label="Conversation", height=300)
                with gr.Row():
                    msg      = gr.Textbox(placeholder="Type your message…", show_label=False, container=False)
                    send_btn = gr.Button("Send", variant="primary")
                    stop_btn = gr.Button("Stop")
                    clr_btn  = gr.Button("Clear")

                gr.Examples(examples, inputs=[msg], label="Click example, then Send")

            # — Middle: Agent’s Thought/Action Log —
            with gr.Column(scale=1):
                log_md   = gr.Markdown("### Agent’s Reasoning Log", label="Logs", height=600)

            # — Right: Actions / Shopping Cart —
            with gr.Column(scale=1):
                cart_md  = gr.Markdown("### 🛒 Your Actions / Cart", label="Cart", height=600)

        # === Events ===

        # 1) Build index when button clicked
        build_btn.click(
            fn=build_index,
            inputs=[video_file],
            outputs=[status],
            queue=True
        )

        # 2) Chat send or Enter → handle_user_message → RAG search → run_fn → update chat/log/cart
        def _chain_event(trigger):
            return (
                trigger(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False)
                       .then(search, [chatbot], [image_list, txt_list], queue=True)
                       .then(run_fn,  [chatbot, log_md, image_list, txt_list], [chatbot, log_md, cart_md], queue=True)
            )

        submit_event = _chain_event(msg.submit)
        click_event  = _chain_event(send_btn.click)

        # 3) Stop button cancels the above
        stop_btn.click(
            fn=stop_fn,
            inputs=None,
            outputs=None,
            cancels=[submit_event, click_event],
            queue=False
        )

        # 4) Clear conversation & cart
        clr_btn.click(
            fn=lambda: ( "", [], "### Agent’s Reasoning Log", "### 🛒 Your Actions / Cart" ),
            inputs=None,
            outputs=[msg, chatbot, log_md, cart_md],
            queue=False
        )

    return demo
