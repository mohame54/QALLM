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


aya_start_turn_token = "<|START_OF_TURN_TOKEN|>"
aya_end_turn_token = "<|END_OF_TURN_TOKEN|>"
aya_system_token = "<|SYSTEM_TOKEN|>"
aya_user_token = "<|USER_TOKEN|>"
aya_chatbot_token = "<|CHATBOT_TOKEN|>"
aya_response_st = "<|START_RESPONSE|>"
aya_response_end = "<|END_RESPONSE|>"
aya_bos_token = "<BOS_TOKEN>"


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


def strip_qwen_thinking_tokens(text: str, question="") -> str:
    """Remove <think>…</think> blocks emitted by Qwen reasoning models."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text =  text.strip()
    if question:
        text = text.replace(question, "")
    return text

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


class AyaChatPromptTemplate:
    def __init__(
        self,
        tokenizer,
        system_prompt: str = SYSTEM_PROMPT,
        bos_token: str = aya_bos_token,
        start_turn_token: str = aya_start_turn_token,
        end_turn_token: str = aya_end_turn_token,
        system_token: str = aya_system_token,
        user_token: str = aya_user_token,
        chatbot_token: str = aya_chatbot_token,
        response_st: str = aya_response_st,
        response_end: str = aya_response_end,
        padding_side: str = "left",
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.roles_to_tokens = {
            "system": aya_system_token,
            "user": aya_user_token,
            "assistant": aya_chatbot_token,
        }
        self.bos_token = bos_token
        self.start_turn_token = start_turn_token
        self.end_turn_token = end_turn_token
        self.system_token = system_token
        self.user_token = user_token
        self.chatbot_token = chatbot_token
        self.response_st = response_st
        self.response_end = response_end
        self.padding_side = tokenizer.padding_side

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def set_bos_token(self, bos_token: str):
        self.bos_token = bos_token

    def set_start_turn_token(self, start_turn_token: str):
        self.start_turn_token = start_turn_token

    def apply_chat_prompt_template(
        self,
        messages: list[dict[str, str]],
        generation: bool = False
    ) -> str:
        txt = aya_bos_token

        # Inject only your system prompt
        txt += self.start_turn_token + self.system_token + self.system_prompt + self.end_turn_token

        for msg in messages:
            if msg["role"] == "system":
                continue  # skip any system message in the list, already handled above
            token = self.roles_to_tokens.get(msg["role"])
            content = msg["content"]
            if msg["role"] == "assistant":
                content = self.response_st + content + self.response_end
            txt += self.start_turn_token + token + content + self.end_turn_token

        if messages[-1]["role"] == "user" and generation:
            token = self.roles_to_tokens.get("assistant")
            txt += self.start_turn_token + token + self.response_st

        return txt

    def create_token_labels(
        self,
        messages: list[dict[str, str]],
    ):
        inps_full_txt = self.apply_chat_prompt_template(messages)
        inps_full_ids = self.tokenizer.encode(inps_full_txt, add_special_tokens=False)

        inps_prefix_txt = self.apply_chat_prompt_template(messages[:-1])
        inps_prefix_ids = self.tokenizer.encode(inps_prefix_txt, add_special_tokens=False)
        
        
        # 3. Assistant portion in token space
        assistant_tokens = inps_full_ids[len(inps_prefix_ids):]
        marker_ids = self.tokenizer.encode(self.response_st, add_special_tokens=False)

        r = None
        for i in range(len(assistant_tokens) - len(marker_ids) + 1):
            if list(assistant_tokens[i:i + len(marker_ids)]) == marker_ids:
                r = len(inps_prefix_ids) + i + len(marker_ids)
                break

        if r is None:
            # Debug output so you can see exactly what's happening
            raise ValueError(
                f"couldn't find '{self.response_st}' in assistant tokens.\n"
                f"marker_ids={marker_ids}\n"
                f"assistant_tokens={assistant_tokens}\n"
                f"assistant decoded='{self.tokenizer.decode(assistant_tokens)}'"
            )

        labels = [-100] * len(inps_full_ids)
        labels[r:] = inps_full_ids[r:]
        attention_mask = [1] * len(inps_full_ids)
        return inps_full_ids, attention_mask, labels

    
    def stop_token_ids(self):
        stop_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.encode("<|END_RESPONSE|>", add_special_tokens=False)[0],
            self.tokenizer.encode("<|END_OF_TURN_TOKEN|>", add_special_tokens=False)[0],
        }
        stop_ids = {x for x in stop_ids if x is not None}
        return list(stop_ids)
        
