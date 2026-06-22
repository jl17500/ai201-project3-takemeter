"""
TakeMeter — WSB Discourse Classifier
Stretch feature: deployed Gradio interface.

Loads the fine-tuned distilbert-base-uncased checkpoint and classifies
a pasted WSB post into one of three labels:
  - due_diligence
  - hot_take
  - reaction

Usage:
    pip install gradio transformers torch
    python takemeter_app.py

Place your saved model directory at ./model/ (the folder you downloaded
from Colab containing config.json, pytorch_model.bin or model.safetensors,
tokenizer files, etc.). If no local model is found, the app falls back to
the base distilbert-base-uncased weights as a placeholder so the interface
still loads — predictions will be random in that case.
"""

import gradio as gr
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Model config ──────────────────────────────────────────────────────────────

LABEL_MAP = {
    0: "due_diligence",
    1: "hot_take",
    2: "reaction",
}

LABEL_DESCRIPTIONS = {
    "due_diligence": "Research-backed argument citing specific financial evidence.",
    "hot_take":      "Bold confident claim with little to no supporting evidence.",
    "reaction":      "Emotional response to a market event or personal outcome.",
}

LABEL_COLORS = {
    "due_diligence": "#00C896",   # terminal green
    "hot_take":      "#F5A623",   # amber
    "reaction":      "#E84040",   # red
}

MODEL_DIR = "./model"
FALLBACK_MODEL = "distilbert-base-uncased"
CONFIDENCE_WARNING_THRESHOLD = 0.40

# ── Load model ────────────────────────────────────────────────────────────────

def load_model():
    source = MODEL_DIR if os.path.isdir(MODEL_DIR) else FALLBACK_MODEL
    using_fallback = source == FALLBACK_MODEL

    tokenizer = AutoTokenizer.from_pretrained(source)

    if using_fallback:
        # Load with a fresh classification head (random weights) so the
        # interface renders — outputs will be meaningless.
        model = AutoModelForSequenceClassification.from_pretrained(
            source,
            num_labels=3,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(source)

    model.eval()
    return tokenizer, model, using_fallback


tokenizer, model, USING_FALLBACK = load_model()


# ── Inference ─────────────────────────────────────────────────────────────────

def classify(post_text: str):
    if not post_text or not post_text.strip():
        return (
            "—",
            "Paste a WSB post above and click **Classify**.",
            {label: 0.0 for label in LABEL_MAP.values()},
        )

    inputs = tokenizer(
        post_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    pred_label = LABEL_MAP[pred_idx]
    confidence = probs[pred_idx].item()

    # Build confidence dict for the bar chart (label → probability)
    confidence_dict = {LABEL_MAP[i]: round(probs[i].item(), 4) for i in range(3)}

    # Verdict text
    low_confidence = confidence < CONFIDENCE_WARNING_THRESHOLD
    verdict_md = f"""
### Verdict: `{pred_label}`

**Confidence:** {confidence:.1%}{"  ⚠️ *Below 0.40 — treat as uncertain*" if low_confidence else ""}

_{LABEL_DESCRIPTIONS[pred_label]}_
"""

    if USING_FALLBACK:
        verdict_md += "\n\n> ⚠️ **No fine-tuned model found at `./model/`** — running on base DistilBERT weights. Predictions are random. Download your Colab checkpoint and place it at `./model/` to use the real classifier."

    return pred_label, verdict_md, confidence_dict


# ── Example posts ────────────────────────────────────────────────────────────

EXAMPLES = [
    ["PYPL FCF yield is 9% at current price. Active accounts down 4% YoY but transactions per account up 11%. New management buying back $5B through 2025. Cheapest large-cap tech stock by FCF I can find right now. Long calls dated June."],
    ["NVIDIA is the next Cisco. We're at the top, mark my words. $5000 puts."],
    ["Lost 80% of my port on SPY puts today. My wife doesn't know yet. I'm done with options forever."],
    ["NVDA's P/E is now 75 vs sector median 32. Q3 datacenter revenue grew 206% YoY but Q4 guidance was conservative at +15%. Margins are at all-time highs which historically signals a top. Stock is priced for perfection, any guidance miss takes it back to $400."],
    ["Gold is done. (for now) TL;DR: Rates are staying high. Gold priced in continued inflation + rate cuts + financial doom. Warsh isn't cutting and 10y yield is increasing. Will continue to fall."],
]


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Root palette ── */
:root {
    --bg:        #0D1117;
    --surface:   #161B22;
    --border:    #30363D;
    --green:     #00C896;
    --amber:     #F5A623;
    --red:       #E84040;
    --text:      #E6EDF3;
    --muted:     #8B949E;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    --font-sans: 'Inter', 'Helvetica Neue', sans-serif;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
}

/* Header */
#takemeter-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
}
#takemeter-header h1 {
    font-family: var(--font-mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--green);
    letter-spacing: -0.02em;
    margin: 0 0 0.2rem 0;
}
#takemeter-header p {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0;
    font-family: var(--font-mono);
}

/* Textarea */
textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
}
textarea:focus {
    border-color: var(--green) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(0,200,150,0.15) !important;
}

/* Classify button */
#classify-btn {
    background: var(--green) !important;
    color: #000 !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    letter-spacing: 0.03em;
    transition: opacity 0.15s;
}
#classify-btn:hover { opacity: 0.85 !important; }

/* Verdict panel */
#verdict-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
    font-family: var(--font-sans) !important;
    min-height: 120px;
}
#verdict-panel h3 { margin-top: 0; color: var(--green); font-family: var(--font-mono); }

/* Bar chart */
#conf-chart .bar { background: var(--green) !important; }

/* Label chips in examples */
.gr-samples-table { font-family: var(--font-mono) !important; font-size: 0.8rem !important; }

/* Misc */
label, .label-wrap span { color: var(--muted) !important; font-size: 0.78rem !important; font-family: var(--font-mono) !important; }
"""

with gr.Blocks(css=CSS, title="TakeMeter") as demo:

    gr.HTML("""
    <div id="takemeter-header">
        <h1>▶ TakeMeter</h1>
        <p>WSB discourse classifier · distilbert-base-uncased fine-tuned · AI201 Project 3</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            post_input = gr.Textbox(
                label="Paste a WSB post",
                placeholder='e.g. "PYPL FCF yield is 9% at current price. Active accounts down 4% YoY..."',
                lines=7,
                max_lines=20,
            )
            classify_btn = gr.Button("Classify →", elem_id="classify-btn")

            gr.Examples(
                examples=EXAMPLES,
                inputs=post_input,
                label="Example posts",
            )

        with gr.Column(scale=2):
            verdict_output = gr.Markdown(
                value="_Paste a post and click **Classify**._",
                elem_id="verdict-panel",
            )
            confidence_chart = gr.Label(
                label="Confidence by class",
                num_top_classes=3,
                elem_id="conf-chart",
            )

    # Hidden label output (used for the verdict color — not displayed directly)
    label_output = gr.Textbox(visible=False)

    classify_btn.click(
        fn=classify,
        inputs=post_input,
        outputs=[label_output, verdict_output, confidence_chart],
    )

    gr.HTML("""
    <div style="margin-top:2rem; padding-top:1rem; border-top:1px solid #30363D;
                font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#8B949E;">
        ⚠ Predictions with confidence &lt; 0.40 are near-random — treat as uncertain.<br>
        Labels: <span style="color:#00C896">due_diligence</span> ·
                <span style="color:#F5A623">hot_take</span> ·
                <span style="color:#E84040">reaction</span>
    </div>
    """)


if __name__ == "__main__":
    demo.launch()