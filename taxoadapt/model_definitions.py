"""
model_definitions.py

Modified to support a FREE local Ollama workflow (no OpenAI billing, no vLLM).

How to use:
- Set args.llm = "ollama"
- Ensure Ollama is running and you have pulled the model:
    ollama pull llama3.2:3b
- Ollama OpenAI-compatible endpoint is:
    http://localhost:11434/v1
"""

import numpy as np
import os
from tqdm import tqdm
from openai import OpenAI

openai_key = os.getenv("OPENAI_API_KEY")


# map each term in text to word_id
def get_vocab_idx(split_text: str, tok_lens):
    vocab_idx = {}
    start = 0

    for w in split_text:
        if w not in vocab_idx:
            vocab_idx[w] = []
        vocab_idx[w].extend(np.arange(start, start + len(tok_lens[w])))
        start += len(tok_lens[w])

    return vocab_idx


def get_hidden_states(encoded, data_idx, model, layers, static_emb):
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    # Only select the tokens that constitute the requested word
    for w in data_idx:
        static_emb[w] += output[data_idx[w]].sum(dim=0).cpu().numpy()


def chunkify(text, token_lens, length=512):
    chunks = [[]]
    split_text = text.split()
    count = 0
    for word in split_text:
        new_count = count + len(token_lens[word]) + 2  # 2 for [CLS] and [SEP]
        if new_count > length:
            chunks.append([word])
            count = len(token_lens[word])
        else:
            chunks[len(chunks) - 1].append(word)
            count = new_count

    return chunks


def constructPrompt(args, init_prompt, main_prompt):
    # GPT and Ollama both use chat-style messages
    if args.llm in ("gpt", "ollama"):
        return [
            {"role": "system", "content": init_prompt},
            {"role": "user", "content": main_prompt},
        ]
    # Any other non-chat backend would use plain string prompts
    return init_prompt + "\n\n" + main_prompt


def initializeLLM(args):
    """
    Initializes the chosen LLM backend.

    Supported:
    - args.llm == "ollama"  -> free local model via Ollama (recommended)
    - args.llm == "gpt"     -> OpenAI API (paid)
    """
    args.client = {}

    if args.llm == "gpt":
        # Paid OpenAI path
        args.client["gpt"] = OpenAI(api_key=openai_key)

    if args.llm == "ollama":
        # Free local path
        args.client["ollama"] = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # dummy key, Ollama ignores it
        )
        # You can override this from main.py by adding an argparse arg later if you want
        if not hasattr(args, "ollama_model") or args.ollama_model is None:
            args.ollama_model = "llama3.2:3b"

    return args


def promptGPT(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
    outputs = []
    for messages in tqdm(prompts):
        if json_mode:
            response = args.client["gpt"].chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                stream=False,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
        else:
            response = args.client["gpt"].chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                stream=False,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
        outputs.append(response.choices[0].message.content)
    return outputs


def promptOLLAMA(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
    """
    Calls a local Ollama model via its OpenAI-compatible API.

    NOTE:
    - If you get an error related to response_format/json_object, rerun with json_mode=False
      (some Ollama builds/models are stricter about JSON formatting).
    """
    outputs = []
    model_name = getattr(args, "ollama_model", "llama3.2:3b")

    for messages in tqdm(prompts):
        if json_mode:
            response = args.client["ollama"].chat.completions.create(
                model=model_name,
                stream=False,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
        else:
            response = args.client["ollama"].chat.completions.create(
                model=model_name,
                stream=False,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )

        outputs.append(response.choices[0].message.content)

    return outputs


def promptLLM(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
    if args.llm == "gpt":
        return promptGPT(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)
    if args.llm == "ollama":
        return promptOLLAMA(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)

    raise ValueError(f"Unsupported llm backend: {args.llm}. Use 'ollama' (free) or 'gpt'.")