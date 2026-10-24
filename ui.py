import os

import gradio as gr
import requests

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/ask")


def ask_question(question: str) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "Please enter a question.", ""

    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        return f"Could not reach the question-answering API: {exc}", ""

    sources = "\n\n".join(
        f"Page {source['page']}: {source['content']}"
        for source in result.get("sources", [])
    )
    return result["answer"], sources


with gr.Blocks(title="PDF Question Answering") as demo:
    gr.Markdown("# Ask a question about the PDF")
    question = gr.Textbox(
        label="Your question",
        placeholder="Type a question and click Ask",
        lines=2,
    )
    ask_button = gr.Button("Ask")
    answer = gr.Textbox(label="Answer", lines=5)
    sources = gr.Textbox(label="Sources", lines=8)

    ask_button.click(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources],
    )
    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=[answer, sources],
    )

demo.launch()