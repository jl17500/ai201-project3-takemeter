# TakeMeter (planning.md)

---

## Community

`r/wallstreetbets` as source community for this project.

Had considered additional examples from r/stock and r/investing to balance out the `due_diligence` but decided against it as the three subs had noticebly diff house style. If i were to train on a mix ed corpus the model could end up learning subreddit style as a proxy for `due_diligence` rather than the actual underlying distinction. 

WSB is good fit for this kind of classification task because:

1. Its range of quality is huge. Typical hot page on WSB has everything from real DD posts that breaks down filings or competitor margins to single YOLO calls like "$5k calls on NVDA, see you on the moon," to straight up memes about losing money. So the spread is pretty wide and that allows the boundaries between the labels actually meaningful.

2. Community already thinks in categories. WSB has post flairs for `DD`, `YOLO`,`Loss`,`Gain`,`Discussion`. So Users already self categorize along those lines, meaning that my labels aren't smt to impose from outside as they map onto the thing the community already uses.

3. Post vary in length and form. Some sentences, others paragraphs and sometimes with screenshots. Classifier can handle that range, which makies evaluation more diagnostic.

Why this matter to me? I occasionally go on WSB and r/stocks and there's a difference between someone who did research and someone just yolo posting their meme stock tanking. Therefore a classifier that surfaces DD when markets are quiet and reactions during news events would be quite useful as a feed filter. 

---

## Label Taxonomy

3 labels that are mutually exclusive and exhaustive enough to cover most WSB posts without needing an `other` bucket. 

**Label 1:** `due_diligence`

**Def:** research backed argument that cites specific financial evidence (earnings, ratios, comp dynamics, filings, historical) and actually uses the evidence to support a claim about a stock or market. 

**Key Signal:** specific evidence doing the argument. Post can be wrong and still be `DD` if argues from facts. 

**Example 1:**
> "NVDA's P/E is now 75 vs sector median 32. Q3 datacenter revenue grew 206% YoY but Q4 guidance was conservative at +15%. Margins are at all-time highs which historically signals a top. Stock is priced for perfection, any guidance miss takes it back to $400."
 
**Example 2:**
> "PYPL FCF yield is 9% at current price. Active accounts down 4% YoY but transactions per account up 11%. New management buying back $5B through 2025. Cheapest large-cap tech stock by FCF I can find right now. Long calls dated June."
 
---

**Label 2:** `hot_take`

**Def:** Bold confident claim about a stock or the market with little to no specific supoorting evidence. May have a fact or two but mainly asserts rather than argues.

**Key Signal:** confidence overcoming the evidence. Decorative evidence doesn't make the post `DD` if evidence isn't driving the argument.

**Example 1:**
> "NVIDIA is the next Cisco. We're at the top, mark my words. $5000 puts."
 
**Example 2:**
> "Apple is dead. R&D hasn't grown in 3 years. They'll be irrelevant by 2027."
 
---

**Label 3:** `reaction`

**Def:** Emotional response to market event or personal trading outcome. Express feeling and doesn't attempt an argument. 

**Key Signal:** post exists to share an emotion not to inform or persuade. If delete emotion, post would pretty much have nothing left.

**Example 1:**
> "Lost 80% of my port on SPY puts today. My wife doesn't know yet. I'm done with options forever."
 
**Example 2:**
> "FED PIVOT INCOMING. We're so back. SPY $700 by year end LFG. TO THE MOONNNNNNNNN"🚀🚀🚀
 
---

**Hardest Anticipated Edge Case**

The hardest case sits between `due_diligence` and `hot_take` where the post cites one cherry picked stat to cover the opinion.

Example:
> AAPL is overvalued. P/E of 32 on a company that hasn't grown revenue in 2 years. Easy short.

Now the evidence is valid but it isn't doing any argunmentative work (it's decoration). Real `DD` would test the P/E against historical context, peers, growth expectations. Post above just asserts the conclusion with 1 number placed in the front. 

**Decision Rule**: If cited evidence would still support the claim after removed opinion framing, it's `DD`. If evidence is 1 or 2 cherry picked stats and would read almost identically as pure opinion if deleted stats, it's `hot_take`.

Another edge case could be for emotional reactions that have a brief opinion stuffed in

> Down $40K on TSLA puts today. This company is obviously a fraud and the market will figure it out eventually.


If the dominant function of the post is to sahre a feeling, in this case rage vindication, label it `reaction` even if there's a small argument tacked on.

---

**Data Collection Plan**

r/wallstreetbets only, top + hot from last 90 days

Method: Two phase Collection:

1. Initial Pass, manual copy to CSV. 
2. Scrape.  After intial pass it was obvious from skimming titles that `due_diligence` was gping to come in underrepresented(WSB's hot page is dominated by reactions). Used Gemini to write a scraper for additional WSB `DD` flair posts. 

Target distribution: ~70 examples per label so ~210 total minimum. Current dataset at around 279 rows prelabeling, so headrooms to discard any rows that turn out to be unusable. 


| Column | What it holds |
|---|---|
| `text` | Post title + body concatenated. URLs and images stripped. |
| `label` | One of `due_diligence`, `hot_take`, `reaction`. |
| `notes` | Free-form notes — Claude's one-line reason during pre-labeling, plus any borderline calls I want to revisit. |
| `pre_labeled` | `no` / `pending` / `yes` — tracks whether the row was pre-labeled by Claude (`pending`) and whether I have reviewed it (`yes`). |
| `subreddit` | Source subreddit. Will be `wallstreetbets` for every row given the single-community decision; kept for auditability. |
| `url` | Source URL. |

What to do if label is underrepresented after labeling. Target that label specifically using flair filters. No single label can go past 70% of dataset. 

Before I commit labeling, read 30 to 40 post from WSB and see if I can apply my labels to these without any new labeling popping up. If new edge case shows up I'll revise def before annotating. 

---

## Evaluation Metrics

Overall Acc - 33% is random for 3 balanced classes so above that is signal

Macro F1 (primary) - average of per class F1s. Catches the 'just predicts the majority class" case that raw accuracy can't see

Per class precision and recall - direction of error. High precision and low recall on `due_dilligence` = too conservative. Low and high over labels DD.

Confusion Matrix - which boundary model can't learn. `due_diligence` and `hot_take` is diff story from `hot_take` and `reaction`.

Acc alone hides imbalance and direction of error so need per class breakdown to know what kind mistake model is making.

---

## Definition of Success

3 criteria on test set:

- Macro F1 >= 0.70 Achievable with 200 examples + DistilBERT but requires consistent labels and learnable boundaries. 

- NO single class with F1<0.55. Below that means the model failed to learn one boundary entirely. Labeling and data problem not tuning one.

- Beats Groq zero shot by >= 5 macro F1 points. Otherwise finetuning added no value.

Stretch: macro F1>=0.8. Deployment threshold >=0.75 and below that too many DD post is filtered as `hot_take` and filter just frustrates users.

---

## AI Tool Plan

1. Label stress testing
- Paste my Def + decision rules to Claude, ask for 10 boundary posts (5 DD/hot-take, 5 hot-take/reaction). If I can't label correctly then my definition too vague

2. Annotation assistance
- I'll pre label batches of around 25 post using Claude. Each batch gets my labeled defs. 12 example post per label then 25 unlabled post with a respond with just the label per post instruction.

- Every prelabel is reviewerd by me before going into dataset
- Anytime I disagree with Claude's pre label, write the reason in notes column so I can find disagreement patterns later.
- If I catch myself rubber stampoing more than a few inb a row without thinking, stop and reread the spec, as skimming defeats the purpose.

3. Failure Pattern Analysis
- After fine tuning I'll paste the misclassified test exaqmples with true label, predicted label, and confidence into Claude then ask: What patterns do yoiu see across these errors? Look for label pair confusions, post length effects, sarcasm, stylistic signals. 

- If CLaude claims a pattern I can't confirm, goes into "claims I couldn't verify" subsection so the reader sees what I rejected
- I dont let Claude write analysis seection.

---

## Stretch Features

Confidence Calibration - does model confidence track its actual accuracy? Plan: bin test predictions by confidence (0.5-0.6,0.6-0.7,0.7-0.8 and so on), report accuracy per bin, plot a reliability diagram. 

Error Pattern Analysis - Other than individual errors, identify one systematic pattern with supporting evidence from error set

Deployed Gradio Interface - Gradio app that takes pasted post and outputs the predict label and bar chart of all 3 class confidences. 
