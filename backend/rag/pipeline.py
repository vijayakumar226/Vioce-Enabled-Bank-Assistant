import json
import logging
import os
import threading

import faiss
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "indexes", "rag_index")
)
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# What was wrong: retrieval was limited to 3 chunks, which could miss relevant supporting passages. What changed: fetch 5 chunks for each query.
TOP_K = 5
# What was wrong: the similarity cutoff was too strict at 0.45 and filtered out relevant chunks. What changed: lowered and renamed the cutoff to SIMILARITY_THRESHOLD = 0.30.
SIMILARITY_THRESHOLD = 0.30
SYSTEM_PROMPT = """You are an SBI banking assistant.

CRITICAL FORMATTING RULES — ALWAYS FOLLOW:

RULE A: START every answer with an emoji + topic heading.
Example: ️ Loans Available in SBI

RULE B: FOR LISTS OF ITEMS (loans, products, documents,
account types) — use this format with a blank line
between each item:

  1. [Item Name]
     [One sentence description]

  2. [Item Name]
     [One sentence description]

RULE C: FOR MULTIPLE WAYS/METHODS/CHANNELS — use:

  Option 1: [Method Name]
  • Step: [what to do]
  • Need: [what is required]

  Option 2: [Method Name]
  • Step: [what to do]
  • Need: [what is required]

RULE D: FOR CHARGES/AMOUNTS — use:

  • [Item]: ₹[amount] + GST
  • [Item]: ₹[amount] + GST
  • [Item]: Free

RULE E: END every answer with a tip or contact:
   Tip: [useful advice]
  OR
   Help: Call 1800 425 3800

RULE F: NEVER write answers as paragraphs.
NEVER combine multiple items on one line.
ALWAYS put a blank line between each item.
NEVER say "based on context" or mention documents.

RULE G: FOR GREETINGS (hello, hi, how are you):
   Hello! I am your SBI Banking Assistant.
  I can help with loans, accounts, charges,
  complaints and more.
  What would you like to know today?

RULE H: FOR SIMPLE YES/NO FACTS:
  ✅ [Direct answer in one line]
  Then add relevant details below if needed.

══════════════════════════════════════════════
EXAMPLES — FOLLOW THESE EXACTLY:
══════════════════════════════════════════════

EXAMPLE 1 — List of products:
Q: What loans are available in SBI?

️ Loans Available in SBI

1. SBI Home Loan Scheme
   For purchasing or constructing a house or flat.

2. SBI Pre-approved Home Loan
   Get approval before selecting a property.

3. SBI Yuva Home Loan
   Designed for young salaried borrowers.

4. SBI Max Gain Home Loan
   Overdraft facility to reduce interest burden.

5. SBI Realty Home Loan
   For buying a plot of residential land.

6. SBI NRI Home Loan
   For Non-Resident Indians buying property in India.

7. SBI Gram Niwas / Sahyog Niwas
   For rural and tribal area housing needs.

8. SBI Green Home Loan
   For eco-friendly and green certified homes.

 Tip: Call 1800 425 3800 or visit www.sbi.co.in
for current interest rates and eligibility.

──────────────────────────────────────────────

EXAMPLE 2 — Multiple ways/methods:
Q: How do I open a savings account?

 Ways to Open an SBI Savings Account

Option 1: YONO App (Digital)
- Process: Download YONO → Fill details online
  → Visit branch once to complete.
- Need: Must be 18+ with a valid mobile number.

Option 2: Branch Visit
- Process: Walk into any SBI branch directly.
- Need: Original Aadhaar card and PAN card.

Option 3: Small Account
- Process: Visit branch for simplified setup.
- Need: Basic ID proof only.

 Tip: Carry a passport photo and Aadhaar
when visiting the branch.

──────────────────────────────────────────────

EXAMPLE 3 — Charges list:
Q: What are ATM withdrawal charges?

 SBI ATM Withdrawal Charges

At SBI ATMs:
- First 5 transactions per month: Free
- Beyond 5 transactions: ₹21 + GST each

At Other Bank ATMs:
- First 3 transactions per month: Free
- Beyond 3 transactions: ₹15 + GST each

Non-financial transactions:
- Beyond free limit: ₹10 + GST each

⚠️ Transaction declined due to low balance:
- Charge: ₹20 + GST per attempt

 Help: Call 1800 425 3800 for card issues.

──────────────────────────────────────────────

EXAMPLE 4 — Simple fact:
Q: Is home loan prepayment free?

✅ Yes — SBI charges NO prepayment penalty
on home loans.

This applies to both fixed and floating rate
loans, regardless of the source of funds.

 Tip: You can prepay any amount at any time
without any extra charges.

──────────────────────────────────────────────

EXAMPLE 5 — Complaint process:
Q: How do I complain to SBI?

 How to File a Complaint with SBI

Option 1: Branch Visit
- Walk in and ask for the complaint book.
- Resolution: Within 10 days.
- You get an SMS with your complaint number.

Option 2: Phone (Toll Free 24x7)
- Call: 1800 425 3800 or 1800 11 22 11

Option 3: SMS
- Send: UNHAPPY to 8008202020

Option 4: Email
- Write to: contactcentre@sbi.co.in

Option 5: Online
- Visit: www.sbi.co.in → Customer Care

⏱️ Resolution Timeline:
- Branch handles: Days 1 to 10
- Local Head Office: Days 11 to 15
- Corporate Centre: Days 16 to 21
- Maximum total time: 21 days

⚠️ Not resolved in 30 days? Approach the
Banking Ombudsman at your nearest SBI branch.

══════════════════════════════════════════════
ANSWER RULES SUMMARY:
══════════════════════════════════════════════
- List of things → numbered list with blank lines
- Multiple ways → Option 1 Option 2 with bullets
- Charges → bullet list with ₹ amounts
- Simple fact → one line with ✅
- Always end with  tip or  contact
- Never write paragraphs
- Never put two items on the same line
- Always leave blank line between items
- Use exact amounts from the documents
"""
SMALL_TALK = {
    "hello": "Hello! I'm your banking assistant. How can I help you today?",
    "hi": "Hi there! How can I assist you with your banking needs today?",
    "hey": "Hey! How can I help you today?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! How can I help you today?",
    "good evening": "Good evening! How can I assist you today?",
    "how are you": "I'm doing great, thank you for asking! How can I assist you with your banking needs today?",
    "how are you doing": "I'm doing well! Ready to help you with any banking questions you have.",
    "how do you do": "I'm here and ready to help! What banking question can I answer for you?",
    "who are you": "I'm an AI-powered banking assistant. I can help you with loans, accounts, KYC, interest rates, and more.",
    "what are you": "I'm a banking chatbot trained on our bank's documents. Ask me anything about our products and services.",
    "what can you do": "I can answer questions about loans, savings accounts, fixed deposits, KYC requirements, interest rates, and other banking services.",
    "what can you help with": "I can help with loans, savings accounts, fixed deposits, KYC requirements, interest rates, and general banking queries.",
    "thank you": "You're welcome! Is there anything else I can help you with?",
    "thanks": "Happy to help! Let me know if you have any other questions.",
    "thank you so much": "You're most welcome! Feel free to ask if you need anything else.",
    "bye": "Goodbye! Have a great day. Feel free to return if you have any banking questions.",
    "goodbye": "Goodbye! Don't hesitate to reach out if you need any help.",
    "see you": "See you! Have a wonderful day.",
    "good": "Thank you! Let me know if you need any banking assistance.",
    "great": "Glad to hear that! Anything else I can help you with?",
    "ok": "Sure! Feel free to ask if you have any banking questions.",
    "okay": "Alright! Let me know if there's anything else you need.",
}


def detect_small_talk(query: str) -> str | None:
    """
    Check if the query is casual conversation.
    Returns a response string if it matches, None otherwise.
    """
    normalized = query.lower().strip().rstrip("?!.,")

    if normalized in SMALL_TALK:
        return SMALL_TALK[normalized]

    for key, response in SMALL_TALK.items():
        if key in normalized:
            return response

    return None


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

logger.info("pipeline INDEX_DIR=%s", INDEX_DIR)
logger.info("pipeline INDEX_DIR exists=%s", os.path.isdir(INDEX_DIR))

_lock = threading.Lock()
_cache: dict = {
    "index": None,
    "chunks": [],
    "meta": [],
}


def _load_index_from_disk() -> dict:
    idx = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(INDEX_DIR, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {"index": idx, "chunks": chunks, "meta": meta}


def reload_index() -> None:
    """Hot-swap the in-memory FAISS index from disk. Thread-safe."""
    new = _load_index_from_disk()
    with _lock:
        _cache["index"] = new["index"]
        _cache["chunks"] = new["chunks"]
        _cache["meta"] = new["meta"]


reload_index()

embed_model = SentenceTransformer(EMBED_MODEL, local_files_only=True)

hf_token = os.getenv("HF_TOKEN")
logger.info("pipeline HF_TOKEN present=%s", bool(hf_token))
if not hf_token:
    raise RuntimeError(
        "HF_TOKEN not found. Set HF_TOKEN environment variable and restart."
    )
client = InferenceClient(token=hf_token)
logger.info("pipeline Inference client initialized=%s", client is not None)


def _format_source_label(meta: dict) -> str:
    source = meta.get("source", "doc")
    page = meta.get("page")
    row = meta.get("row")
    if page:
        return f"{source}, page {page}"
    if row:
        return f"{source}, row {row}"
    return source


def _build_rag_context(hits: list[tuple[float, str, dict]]) -> tuple[str, list[dict]]:
    context_blocks: list[str] = []
    sources: list[dict] = []

    for score, text, meta in hits:
        if score < SIMILARITY_THRESHOLD:
            continue

        source_label = _format_source_label(meta)
        # What was wrong: only the single strongest result effectively shaped the prompt context. What changed: every retrieved chunk above threshold is passed into the prompt with a source header and explicit separators.
        context_blocks.append(f"[Source: {source_label}]\n{text}")
        sources.append(
            {
                "score": score,
                "text": text,
                "source": meta.get("source", "doc"),
                "page": meta.get("page"),
                "row": meta.get("row"),
                "kind": "rag",
            }
        )

    return "\n\n---\n\n".join(context_blocks), sources


def retrieve(query: str) -> list[tuple[float, str, dict]]:
    q_vec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)

    with _lock:
        idx = _cache["index"]
        chunks = _cache["chunks"]
        meta = _cache["meta"]

    scores, ids = idx.search(q_vec, TOP_K)
    raw_scores = [float(score) for score in scores[0][:3]]
    # What was wrong: retrieval scores were only printed ad hoc and were hard to monitor. What changed: log the top-3 similarity scores at INFO level for every query.
    logger.info("retrieve query=%r top3_scores=%s", query, raw_scores)

    results: list[tuple[float, str, dict]] = []
    for score, idx_value in zip(scores[0], ids[0]):
        if idx_value < 0:
            continue
        results.append((float(score), chunks[int(idx_value)], meta[int(idx_value)]))
    return results


def answer_query(query: str, attachment_contexts: list[dict] | None = None) -> dict:
    query = (query or "").strip()
    if not query:
        return {"answer": "Please type a question.", "sources": []}

    small_talk_response = detect_small_talk(query)
    if small_talk_response:
        logger.info("answer_query query=%r handled_by=small_talk", query)
        return {"answer": small_talk_response, "sources": []}

    hits = retrieve(query)
    rag_context, sources = _build_rag_context(hits)

    attachment_contexts = attachment_contexts or []
    attachment_blocks: list[str] = []
    for attachment in attachment_contexts:
        name = attachment.get("name", "attachment")
        content = (attachment.get("content") or "").strip()
        if not content:
            continue

        attachment_blocks.append(f"[Source: attachment {name}]\n{content}")
        sources.append(
            {
                "score": 1.0,
                "text": content,
                "source": name,
                "page": None,
                "row": None,
                "kind": "attachment",
            }
        )

    context_parts = [part for part in [rag_context, "\n\n---\n\n".join(attachment_blocks)] if part]
    context = "\n\n---\n\n".join(context_parts)

    if not context.strip():
        logger.info("answer_query query=%r context_lines=[]", query)
        return {"answer": "I don't know from the given data.", "sources": []}

    prompt = f"""Context:
{context}

Question: {query}
Answer:"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
    )

    answer = resp.choices[0].message["content"].strip()
    return {"answer": answer, "sources": sources}
