# TakeMeter

Text classifier that reads r/wallstreetbets posts and sorts them into 3 buckets: actual research, bold opinion, and emotional reactions

---

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| GPT-style baseline (Groq, zero-shot) | 0.52 | 0.54 |
| Fine-tuned DistilBERT (this project) | **0.64** | 0.35 |

Sure, Higher accuracy, but worse overall. Fine tuned model got really good at spotting `hot_take` and basically gave up on `reaction`. Baseline was more balanced. Full breakdown in `Evaluation`.

---

## Why r/wallstreetbets

WSB is useful for this as the qualiy of the psot varies quite a bit. (The same front page has someone citing earnings and someone going "5k calls on NVDA, see yall on the moon," and someone positng about them losing their life savings) The range makes label boundaries quite meaningful.

WSB also already uses post flairs such as 'DD', 'YOLO', 'Loss', so the categories aren't going to be invented in that they map onto what the community already thinks of the content.

---

## The Three Labels

### `due_diligence`
Post makes an argument using real financial evidence (evidence, ratios, filings, historical data). Evidence drives the argument and actually supports the claim rather than just decorate. Post can be wrong and still count as 'due_diligence'.

> "NVDA's P/E is now 75 vs sector median 32. Q3 datacenter revenue grew 206% YoY but Q4 guidance was conservative at +15%. Margins are at all-time highs which historically signals a top. Stock is priced for perfection, any guidance miss takes it back to $400."
 
> "PYPL FCF yield is 9% at current price. Active accounts down 4% YoY but transactions per account up 11%. New management buying back $5B through 2025. Cheapest large-cap tech stock by FCF I can find right now. Long calls dated June."
 
### `hot_take`
Confident claim with little to no real supporting evidence. May throw in a stat or two, but the post is mostly just asserting something and not arguing for it.
 
> "NVIDIA is the next Cisco. We're at the top, mark my words. $5000 puts."
 
> "Apple is dead. R&D hasn't grown in 3 years. They'll be irrelevant by 2027."

### `reaction`
Emotional response to something that happened (a loss, a win, a market event). Remove the emotion and nothing left.
 
> "Lost 80% of my port on SPY puts today. My wife doesn't know yet. I'm done with options forever."
 
> "FED PIVOT INCOMING. We're so back. SPY $700 by year end LFG 🚀🚀🚀"
 

### When it's hard to tell
 
**`due_diligence` vs `hot_take`:** Some posts cite one stat to make their opinion sound more credible. 

Rule: if delete stat and still reads the same, it `hot_take`. If the stat is actually doing the work of the argument, it's `due_diligence`.
 
**`reaction` vs `hot_take`:** Some emotional posts have a quick claim tacked on at the end. 

Rule: if the main point is sharing a feeling, it's `reaction` even if there's a sentence of opinion

---
 
## Data
 
**Source:** r/wallstreetbets; top and hot posts from last 90 days.
 
**How it was collected:** Manual copy paste, then scraper `DD`-flair posts because `due_diligence` was way underrepresented in the initial batch.
 
**How it was labeled:** Claude pre-labeled batches of ~25 posts, then every label was reviewed manually.
 
**Label counts** (163 examples total — fell short of the 200 target after filtering image-only, deleted, and link-only posts):
 
| Label | Count | % |
|---|---|---|
| `hot_take` | 110 | 67.5% | (It's below 70%)
| `due_diligence` | 28 | 17.2% |
| `reaction` | 25 | 15.3% |
 
Heavy imbalance but then again WSB really is mostly hot takes and that is also what caused the model's biggest failure [Evaluation]

### Three hard calls
 
1. *"AAPL is overvalued. P/E of 32 on a company that hasn't grown revenue in 2 years. Easy short."* = **`hot_take`**.

 There is a stat but it's not doing any argumentative work so it's decoration. The post is just asserting overvalued, not making a case.

2. *"Down $40K on TSLA puts today. This company is obviously a fraud and the market will figure it out eventually."* = **`reaction`**. 
The fraud claim may sound like opinion, but the whole post is really about expressing rage over a loss. 

3. *"NVIDIA just reported Q1 2026 results. Revenue came in at $81.6 billion, up 85% YoY. Long into earnings, up 12% on my position."* = **`due_diligence`**. Short post, cites a specific verifiable number, draws a direct conclusion. The model got this wrong (predicted `hot_take`)

## Training
 
**Base model:** `distilbert-base-uncased` from HuggingFace  
**Platform:** Google Colab, free T4 GPU
 
Took three attempts to get a model that did anything useful:
 
| Run | What changed | What happened |
|---|---|---|
| Run 1 | Default settings, optimizing for accuracy | Predicted `hot_take` for everything. 68% accuracy, 0.27 macro F1. |
| Run 2 | Added hard class weights to force attention to minority labels | Overcorrected, predicted `due_diligence` for everything. 0.11 macro F1. |
| Run 3 (final) | Softer weights using `sqrt()`, 5 epochs, label smoothing | `hot_take` learned well. `reaction` still failed. 0.35 macro F1. |
 
`hot_take` is 67.5% of training data. Without any correction, model can just predict `hot_take` constantly and get reward. Hard weights fixed that but swung too far. Taking square root of weights compressed the 4:1 ratio to roughly 2:1, which was enough to nudge model without messing it up.
 
**Final settings:**
 
| Setting | Value |
|---|---|
| Epochs | 5 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Weight decay | 0.01 |
| Label smoothing | 0.1 |
| Best checkpoint picked by | Macro F1 (not accuracy) |

 ## Baseline

Used llama-3.3-70b-versatile with no fine-tuning, just a prompt with the label definitions and "respond with only the label name."

Prompt included full label definitions and decision rules from planning.md. All 25 responses parsed cleanly.

---

## Evaluation
 
### Numbers
 
**Baseline (Groq zero-shot):**
 
| Label | Precision | Recall | F1 | Count |
|---|---|---|---|---|
| `due_diligence` | 0.57 | 1.00 | 0.73 | 4 |
| `hot_take` | 0.86 | 0.35 | 0.50 | 17 |
| `reaction` | 0.27 | 0.75 | 0.40 | 4 |
| **macro avg** | 0.57 | 0.70 | **0.54** | 25 |
 
**Fine-tuned DistilBERT:**
 
| Label | Precision | Recall | F1 | Count |
|---|---|---|---|---|
| `due_diligence` | 0.33 | 0.25 | 0.29 | 4 |
| `hot_take` | 0.68 | 0.88 | 0.77 | 17 |
| `reaction` | 0.00 | 0.00 | 0.00 | 4 |
| **macro avg** | — | — | **0.35** | 25 |
 
### Confusion matrix
 
| | Predicted: `due_diligence` | Predicted: `hot_take` | Predicted: `reaction` |
|---|---|---|---|
| **True: `due_diligence`** | 1 | 3 | 0 |
| **True: `hot_take`** | 2 | 15 | 0 |
| **True: `reaction`** | 0 | 4 | 0 |

![Confusion Matrix](confusion_matrix__2_.png) ^if image doesn't work, the above diagram is a pretty accurate representation

Model never predicted `reaction` once. Every `reaction` post got called `hot_take`. 3/4 `due_diligence` posts also got called `hot_take`. The only thing it learned well was `hot_take`.


### Three wrong predictions
 
**1. Actual: `reaction`, Predicted: `hot_take` (confidence: 0.35)**
 
> "Missed the boat, made some money in day trading 0DTE 100% win today! As title says, missed today bull run, but made it today by choose 0DTE within 20 minutes. Beginner/learner of the 0DTE trick..."
 
Someone sharing excitement about lucky trade. No market claim just vibes. Should be `reaction`. Model saw "0DTE," "100% win," and "20 minutes" and pattern matched to `hot_take` because apparently those look like confident assertions. Difference between "I'm excited about what happened to me" and "here's my bold market claim" doesn't show up in the words themselves.
 
**2. Actual: `due_diligence`, Predicted: `hot_take` (confidence: 0.35)**
 
> "Am I the only one confused about why NVIDIA crushed earnings and the stock still went red? And what's with the big jump in Dividend 👍 NVIDIA just reported its Q1 2026 results (fiscal Q1 FY27, quarter e...")
 
Cites real Q1 2026 earnings data, which makes it `due_diligence`. But it opens with "Am I the only one confused...," which reads like an opinion. The model learned that dense analytical prose = `due_diligence`, so a question-framed short post with real data didn't fit its pattern.
 
**3. Actual: `hot_take`, Predicted: `due_diligence` (confidence: 0.35)**
 
> "Gold is done. (for now) TL;DR: Rates are staying high. Gold priced in continued inflation + rate cuts + financial doom. Warsh isn't cutting and 10y yield is increasing. Will continue to fall."
 
Pure `hot_take`. Asserting gold falls because rates stay high, with no actual argument. But it has TLDR header, mentions real concepts (Warsh, 10y yield), sounds like analysis. Model got fooled by format. The post is a case of the "decorative evidence" problem from the label definitions.
 
### What went wrong (the pattern)
 
Every single wrong prediction had a confidence of 0.35 to 0.36. Random chance for 3 classes is 0.33. Model wasn't confidently wrong, it was essentially guessing on everything it got wrong.
 
Two patterns:
 
**Model learned `hot_take` = short + assertive.** That's right often enough (15/17 correct) but also describes `reaction` posts, which are also short and assertive. Problem is expressing a feeling vs. making a market claim doesn't show up in the words.
 
**Model learned `due_diligence` = long analytical.** So short DD posts looked like `hot_take`, and verbose `hot_take` posts with analytical vocabulary looked like `due_diligence`. It's pattern matching and not determining if evidence is doing anything.

## Confidence Calibration

All 9 wrong predictions: confidence 0.35–0.36. All 16 correct predictions: higher than that, which is actually quite useful, as if confidence is below 0.4, prediction is prob wrong or ambiguous. In a real feed filter those should be flagged for human review. 

---

## What model actually learned vs. what was intended

Goal was to classify post by argumentative intent: Does post argue from evidence, assert without evidence, or express feeling?

What the model actually learned: post length and vocabulary.

So now:

- Short + assertive = `hot_take`
- Long + some terms = `due_dligence`
- `reaction` never got learned because it looked similar to `hot_take`

The deeper issue is that reaction was defined by what a post does rather than what it contains. And sure while that distinction is pretty significant for a human who reads WSB, it's not something a model can learn from 25 training examples especially when the words looks relatively the same. Fixable version can maybe try to define reaction as requiring explicit reference to a personal position or outcome.

---

## Did it meet the goals?
 
| Goal | Target | Result |
|---|---|---|
| Macro F1 | ≥ 0.70 | 0.35 ❌ |
| No label with F1 < 0.55 | — | `reaction` F1: 0.00 ❌ |
| Beat baseline by 5+ F1 points | — | Lost by 19 points ❌ |

Disappointingly, all 3 missed. Fine tuned model was worse than the zero shot baseline on every single metric that accounts for minority labels. The only win being raw accuracy, which doesn't even mean much when the dataset is imbalanced. 

---
 
## Deployed interface
 
`interface/app.py` is a Gradio app. Paste a WSB post, get a label, confidence breakdown.
 
```bash
cd interface
pip install gradio transformers torch
python app.py
```
 
Loads fine-tuned model from `./model/`. Predictions below 0.40 confidence are flagged as uncertain.

---

## Spec Reflection

**What helped:**
- Having to write the success criteria before training. "Macro F1 ≥ 0.70, no class below 0.55" made it pretty clear after each run what failed. Without it, accuracy increase from 52% to 64% would've looked like progress when it's not. 

**Diverges:**
- Spec assumed one training run. Took 3...

---
## AI Usage

1. **Pre Labeling**
- Gave my definition + decision rules from `planning.md` to Claude, labeled batches of ~25 posts. Every label was reviewed manually before going into the dataset. About 15% were changed (mostly the `hot_take`/`due_diligence` ones as Claude consistently labeled any post with a financial stat as `due_diligence`).


2. **Failure Pattern Analysis**
- After fine tuning, list of wrong predictions with actual label, predicted label was given to Claude. It correcly identified `reaction`/`hot_take` confusion and flagged short post length as common factor. It also claimed **sarcasm** being a factor, which couldn't confirm from the actual examples so left out.


3. **Debugging class weights** 
Claude helped diagnose why inverse frequency weights in run 2 overcorrected (4:1 ratio too aggressive for 3 epochs on 114 examples) and explained what `sqrt()` would do. Fix worked and so dropped the effective ratio to ~2:1.
