import os
import gdown
from urllib.error import URLError
from urllib.request import Request, urlopen
import re


SYSTEM_PROMPT = (
    "You are a helpful assistant."
    "Answer in the language of the user's question. "
    "Reply with ONLY the final answer text. No thinking, no intro, no outro."
)

# Maps the two-letter country code found in the subset ID to a full country name.
COUNTRY_MAP: dict[str, str] = {
    "Uga": "Uganda",
    "Gha": "Ghana",
    "Eth": "Ethiopia",
    "Ken": "Kenya",
}

STUDENT_TEMPLATE = """
Answer in {language} as spoken in {country}
Question:
{question}"""

TEACHER_TEMPLATE = """
Answer in {language} as spoken in {country}
Question:
{question}

Reference answer (use it to ground your response):
{golden_answer}"""


def strip_qwen_thinking_tokens(text: str) -> str:
    """Remove <think>…</think> blocks emitted by Qwen reasoning models."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def create_question(
    question: str,
    tokenizer,
    answer_start: str = "##Answer:",
    language: str = "",
    country: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    student_template: str = STUDENT_TEMPLATE,
    tokenize:bool = True,
    return_tensors: bool = "pt",
    apply_chat_template=True,
) -> list[dict]:

    kwargs = {
        "tokenize":tokenize,
        "return_tensors":return_tensors
    }
    kwargs = {k:v for k,v in kwargs.items() if v is not None}
    content = (
        student_template.format(question=question, language=language, country=country)
        if language or country
        else question
    )
    inputs = [
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": content},
    ]
    if answer_start is not None:
        inputs.append({"role": "assistant", "content": answer_start})
    if apply_chat_template:
      inputs =  tokenizer.apply_chat_template(
          inputs,
          enable_thinking = False,
          **kwargs
      )
    return inputs


def create_message_instance(
    tokenizer,
    question: str,
    answer:str,
    answer_start: str = "##Answer:",
    language: str = "",
    country: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    student_template: str = STUDENT_TEMPLATE,
    tokenize:bool = True,
    return_tensors: bool = "pt",
    apply_chat_template=False
):
    inputs = create_question(
        question,
        tokenizer,
        None,
        language,
        country,
        system_prompt,
        student_template,
        tokenize,
        return_tensors,
        apply_chat_template
    )
    inputs.append(
        {"role": "assistant", "content": answer_start+ " "+answer}
    )

    return inputs

def create_tokens_labels(
    tokenizer,
    messages,
    answer_start_txt="##Answer:",
    **chat_template_kwargs
):
    chat_template_kwargs = {k: v for k, v in chat_template_kwargs.items() if v is not None}
    chat_template_kwargs['return_tensors'] = None
    chat_template_kwargs['tokenize'] = True

    # 1. Full conversation tokens
    inps_full = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    if isinstance(inps_full, dict):
        tokens = inps_full['input_ids']
        attention_mask = inps_full['attention_mask']
    else:
        tokens = list(inps_full)
        attention_mask = [1] * len(tokens)

    # 2. Prefix tokens (everything up to and including the assistant turn opener)
    prefix_kwargs = {**chat_template_kwargs, 'add_generation_prompt': True}
    inps_prefix = tokenizer.apply_chat_template(messages[:-1], **prefix_kwargs)
    if isinstance(inps_prefix, dict):
        prefix_tokens = inps_prefix['input_ids']
    else:
        prefix_tokens = list(inps_prefix)

    # 3. Assistant portion in token space
    assistant_tokens = tokens[len(prefix_tokens):]

    # 4. Find ##Answer: token IDs *within* the assistant portion
    #    Encode with a leading space/newline context to match how it appears
    #    after the think block
    marker_ids = tokenizer.encode(answer_start_txt, add_special_tokens=False)

    r = None
    for i in range(len(assistant_tokens) - len(marker_ids) + 1):
        if list(assistant_tokens[i:i + len(marker_ids)]) == marker_ids:
            r = len(prefix_tokens) + i + len(marker_ids)
            break

    if r is None:
        # Debug output so you can see exactly what's happening
        raise ValueError(
            f"couldn't find '{answer_start_txt}' in assistant tokens.\n"
            f"marker_ids={marker_ids}\n"
            f"assistant_tokens={assistant_tokens}\n"
            f"assistant decoded='{tokenizer.decode(assistant_tokens)}'"
        )

    labels = [-100] * len(tokens)
    labels[r:] = tokens[r:]

    return tokens, attention_mask, labels
