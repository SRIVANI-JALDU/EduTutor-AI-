 #!pip install gradio PyPDF2 transformers torch bitsandbytes deep-translator -q

import gradio as gr
import torch
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from deep_translator import GoogleTranslator

# === Model Loading ===
try:
    model_name = "ibm-granite/granite-3.3-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", load_in_8bit=True
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    generator = None

# === Model Prompting Function ===
def generate_response(prompt):
    if generator is None:
        return "❌ Model not loaded"
    try:
        out = generator(prompt)
        return out[0]["generated_text"]
    except Exception as e:
        return f"❌ Generation error: {e}"

# === Concept Explanation With Language Translation ===
def concept_understanding(concept, language):
    prompt = f"Explain '{concept}' simply for a 15-year-old with real examples."
    response = generate_response(prompt)
    if language != "English":
        try:
            response = GoogleTranslator(source='auto', target=language.lower()).translate(response)
        except Exception as e:
            return f"⚠ Translation failed: {e}"
    return response

# === PDF Quiz Generator ===
def generate_test_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        if not text:
            return "❌ No text found in PDF."
        prompt = f"""
Make 5 MCQs from this content:

{text}

Format:
Qn: <question>
A. <option A>
B. <option B>
C. <option C>
D. <option D>
Correct Answer: <letter>
"""
        return generate_response(prompt)
    except Exception as e:
        return f"❌ PDF error: {e}"

# === Login Function ===
def login(username, password):
    if username == "admin" and password == "1234":
        return gr.update(visible=True), "✅ Login successful! Please continue below."
    else:
        return gr.update(visible=False), "❌ Invalid credentials. Try again."

# === App UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🔐 *EduTutor AI – Login First*")

    with gr.Row():
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
    login_button = gr.Button("🔓 Login", variant="primary")
    login_status = gr.Textbox(visible=True, interactive=False)

    # After login section
    with gr.Column(visible=False) as main_section:
        gr.Markdown("### 🎓 *EduTutor AI – Personalized Learning with IBM Granite*")
        gr.Markdown("📘 Understand concepts, learn languages, and generate tests from your book/PDF using IBM Granite LLM.")

        concept = gr.Textbox(label="📚 Enter Concept (e.g., Artificial Intelligence)")
        language = gr.Dropdown(["English", "Hindi"], label="🌐 Select Language", value="English")
        pdf_file = gr.File(label="📄 Upload PDF")

        run_btn = gr.Button("🚀 Generate Output", variant="primary")

        concept_output = gr.Textbox(label="💡 Concept Explanation", lines=10, show_copy_button=True)
        quiz_output = gr.Textbox(label="📝 Generated Quiz from PDF", lines=10, show_copy_button=True)

        run_btn.click(
            lambda c, l, p: (concept_understanding(c, l), generate_test_from_pdf(p)),
            inputs=[concept, language, pdf_file],
            outputs=[concept_output, quiz_output]
        )

    login_button.click(fn=login, inputs=[username, password], outputs=[main_section, login_status])

demo.launch(share=True)