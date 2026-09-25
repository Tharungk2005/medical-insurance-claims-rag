from typing import List, Dict, Any, Optional
import tiktoken
from config import MAX_PROMPT_TOKENS

# Tokenizer for context budget tracking
_encoder = None

def get_tokenizer():
    global _encoder
    if _encoder is None:
        try:
            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = False
    return _encoder

def count_tokens(text_or_messages: Any) -> int:
    """Estimate token count using tiktoken cl100k_base."""
    enc = get_tokenizer()
    serialized = str(text_or_messages)
    if enc:
        try:
            return len(enc.encode(serialized))
        except Exception:
            pass
    # Rough fallback: 4 characters per token
    return len(serialized) // 4

# Base System Prompt
SYSTEM_PROMPT = (
    "You are an expert medical insurance claims assistant. Answer user questions strictly and only "
    "using the provided context. If the answer is not directly supported by the context, state clearly: "
    "'I do not have enough information in the available documents to answer this.' "
    "Always cite the specific source document name, section, and page or row number in your response. "
    "Do not extrapolate or speculate beyond the provided medical records, policy clauses, or billing tables."
)

# Realistic Few-Shot Demonstrations
FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": "Why was claim CLM-2024-001 denied for the cardiology consult?"
    },
    {
        "role": "assistant",
        "content": "Based on CLM-2024-001_policy.pdf (Page 2, Section 4.1 'Specialist Consultations'): "
                   "The claim was denied because outpatient specialist consult code 99243 requires prior authorization "
                   "from the primary care physician under Plan B. No prior authorization form was on file at the time of service."
    },
    {
        "role": "user",
        "content": "What is the deductible and coinsurance for in-network inpatient surgery?"
    },
    {
        "role": "assistant",
        "content": "According to policy_summary.pdf (Page 1, 'Inpatient Benefits'): "
                   "The annual in-network individual deductible is $1,500 ($3,000 for families). After the deductible is met, "
                   "in-network inpatient surgery is covered with 20% coinsurance up to the maximum out-of-pocket limit."
    },
    {
        "role": "user",
        "content": "What prescription dosage was authorized in claim CLM-2024-003?"
    },
    {
        "role": "assistant",
        "content": "From CLM-2024-003_prescription.jpg (Rx Scan Report): "
                   "The authorized prescription is Atorvastatin 20mg once daily oral tablet, 90-day supply with 3 refills authorized."
    }
]

COT_SUFFIX = "Think step by step, citing each piece of evidence, before stating your final answer."

def build_context_string(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a structured context block with modality-aware provenance tags.
    """
    if not retrieved_chunks:
        return "No relevant context found."

    context_blocks = []
    for idx, c in enumerate(retrieved_chunks, start=1):
        meta = c.get("metadata", {})
        modality = meta.get("modality", "pdf").lower()
        source = meta.get("source_file", "document")
        page = meta.get("page_number", "N/A")
        claim_id = meta.get("claim_id", "N/A")
        score = c.get("rerank_score", c.get("score", 0.0))

        if modality == "image":
            tag = f"[Source {idx}: Image Scan | File: {source} | Claim: {claim_id} | Confidence: {score:.2f}]"
        elif modality == "table":
            row_idx = meta.get("row_index", "N/A")
            tag = f"[Source {idx}: Billing Table | File: {source} | Row: {row_idx} | Claim: {claim_id} | Confidence: {score:.2f}]"
        else:
            tag = f"[Source {idx}: Document | File: {source} | Page: {page} | Claim: {claim_id} | Confidence: {score:.2f}]"

        context_blocks.append(f"{tag}\n{c.get('text', '').strip()}")

    return "\n\n---\n\n".join(context_blocks)

def build_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    variant: str = "production",
    use_cot: bool = False
) -> List[Dict[str, str]]:
    """
    Assemble the complete prompt message list:
    1. System prompt
    2. Few-shot examples (if variant includes few_shot)
    3. Trimmed conversation history
    4. Modality-tagged context + user query (+ CoT if requested)
    Ensures message list complies strictly with MAX_PROMPT_TOKENS budget.
    """
    if chat_history is None:
        chat_history = []

    # Select variant configuration
    include_few_shot = variant in ["few_shot", "cot", "production"]
    cot_active = use_cot or variant == "cot"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Few-shot examples
    if include_few_shot:
        messages.extend(FEW_SHOT_EXAMPLES)

    # Chat history (take up to last 6 turns)
    recent_history = list(chat_history[-6:])
    messages.extend(recent_history)

    # Context and Query
    context_str = build_context_string(retrieved_chunks)
    query_text = query
    if cot_active:
        query_text = f"{query}\n\n{COT_SUFFIX}"

    user_content = (
        f"Context from medical records, policies, and billing tables:\n\n"
        f"{context_str}\n\n"
        f"Question:\n{query_text}"
    )

    messages.append({"role": "user", "content": user_content})

    # Token Budget Management
    current_tokens = count_tokens(messages)
    if current_tokens > MAX_PROMPT_TOKENS:
        # 1. Trim chat history oldest-first
        while len(recent_history) > 0 and count_tokens(messages) > MAX_PROMPT_TOKENS:
            recent_history.pop(0)
            # Rebuild without old history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if include_few_shot:
                messages.extend(FEW_SHOT_EXAMPLES)
            messages.extend(recent_history)
            messages.append({"role": "user", "content": user_content})

        # 2. If still over budget, truncate last context chunk
        if count_tokens(messages) > MAX_PROMPT_TOKENS and len(retrieved_chunks) > 1:
            trimmed_chunks = retrieved_chunks[:-1]
            trimmed_context = build_context_string(trimmed_chunks)
            user_content = (
                f"Context from medical records, policies, and billing tables:\n\n"
                f"{trimmed_context}\n\n"
                f"Question:\n{query_text}"
            )
            messages[-1] = {"role": "user", "content": user_content}

    return messages

# Production prompt configuration reference
PRODUCTION_PROMPT = {
    "variant": "production",
    "description": "Modality-aware system prompt with 3 grounded medical few-shots and dynamic context injection"
}
