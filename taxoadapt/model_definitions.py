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
import json
from utils import clean_json_string

openai_key = os.getenv("OPENAI_API_KEY")

import hashlib
from pathlib import Path

CACHE_DIR = Path("llm_cache")

def set_cache_dir(path: str):
    global CACHE_DIR
    CACHE_DIR = Path(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key(prompt, model, max_tokens, temperature):
    """Generate a unique cache key from prompt + model settings."""
    content = json.dumps({"prompt": prompt, "model": model, "max_tokens": max_tokens, "temperature": temperature}, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def _load_cache(key):
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None

def _save_cache(key, value):
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


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
    model_name = getattr(args, "ollama_model", "gemma3:12b")

    for messages in tqdm(prompts):
        key = _cache_key(messages, model_name, max_new_tokens, temperature)
        cached = _load_cache(key)
        if cached is not None:
            outputs.append(cached)
            continue

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

        result = response.choices[0].message.content
        _save_cache(key, result)
        outputs.append(result)

    return outputs


def promptLLM(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
    if args.llm == "gpt":
        return promptGPT(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)
    if args.llm == "ollama":
        return promptOLLAMA(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)

    raise ValueError(f"Unsupported llm backend: {args.llm}. Use 'ollama' (free) or 'gpt'.")

def promptLLM_fast(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
    """Uses the fast small model for classification steps (Steps 3 and 4)."""
    original_model = args.ollama_model
    args.ollama_model = args.ollama_model_fast
    result = promptLLM(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)
    args.ollama_model = original_model
    return result

def promptLLM_fast_batched(args, prompts, schema=None, max_new_tokens=2000, json_mode=True, temperature=0.1, top_p=0.99, batch_size=5):
    """
    Batched version of promptLLM_fast — sends multiple prompts in one call.
    Uses batch_size=5 by default for reliability with small models.
    Falls back to individual calls if batch parsing fails.
    """
    import math

    model_name = args.ollama_model_fast
    outputs = []
    n_batches = math.ceil(len(prompts) / batch_size)

    for batch_idx in tqdm(range(n_batches)):
        batch = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # Extract content from each prompt
        batch_contents = []
        for i, msg_list in enumerate(batch):
            user_content = next((m["content"] for m in msg_list if m["role"] == "user"), "")
            batch_contents.append(f"=== PAPER {i+1} ===\n{user_content}")

        system_content = next((m["content"] for m in batch[0] if m["role"] == "system"), "")

        batch_user_content = "\n\n".join(batch_contents)
        batch_user_content += (
            f"\n\nClassify ALL {len(batch)} papers above. "
            f"You MUST return a JSON object with a single key 'results' containing an array of exactly {len(batch)} objects. "
            f"Each object must have keys: dm_type, clinical_task, patient_population, data_modality, clinical_outcome (all boolean). "
            f"Example format: {{\"results\": [{{\"dm_type\": true, \"clinical_task\": true, \"patient_population\": false, \"data_modality\": true, \"clinical_outcome\": false}}, ...]}}"
        )

        batched_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": batch_user_content}
        ]

        cache_key = _cache_key(batched_messages, model_name, max_new_tokens, temperature)
        cached = _load_cache(cache_key)

        batch_parsed = None

        if cached is not None:
            batch_raw = cached
        else:
            try:
                response = args.client["ollama"].chat.completions.create(
                    model=model_name,
                    stream=False,
                    messages=batched_messages,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    timeout=60,  # 60 second timeout per batch
                )
                batch_raw = response.choices[0].message.content
                _save_cache(cache_key, batch_raw)
            except Exception as e:
                print(f"  Batch {batch_idx+1} API error ({e}), falling back to individual calls")
                batch_raw = None

        # Try to parse batch response
        if batch_raw is not None:
            try:
                parsed = json.loads(clean_json_string(batch_raw)) if "```" in batch_raw else json.loads(batch_raw.strip())

                # Handle {"results": [...]} format
                if isinstance(parsed, dict):
                    # Try common wrapper keys
                    for key in ["results", "papers", "classifications", "items", "output"]:
                        if key in parsed and isinstance(parsed[key], list):
                            batch_parsed = parsed[key]
                            break
                    # If still not found, try any list value
                    if batch_parsed is None:
                        for v in parsed.values():
                            if isinstance(v, list) and len(v) == len(batch):
                                batch_parsed = v
                                break
                elif isinstance(parsed, list):
                    batch_parsed = parsed

                # Validate length
                if batch_parsed is not None and len(batch_parsed) != len(batch):
                    print(f"  Batch {batch_idx+1}: got {len(batch_parsed)} results for {len(batch)} papers, falling back")
                    batch_parsed = None

            except Exception as e:
                print(f"  Batch {batch_idx+1} parse error ({e}), falling back to individual calls")
                batch_parsed = None

        # Success — add batch results
        if batch_parsed is not None:
            for item in batch_parsed:
                outputs.append(json.dumps(item))
            continue

        # Fallback: process each paper individually with timeout
        print(f"  Batch {batch_idx+1}: processing {len(batch)} papers individually")
        for msg_list in batch:
            single_key = _cache_key(msg_list, model_name, 500, temperature)
            single_cached = _load_cache(single_key)
            if single_cached is not None:
                outputs.append(single_cached)
                continue
            try:
                response = args.client["ollama"].chat.completions.create(
                    model=model_name,
                    stream=False,
                    messages=msg_list,
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=500,
                    timeout=30,  # 30 second timeout per individual call
                )
                result = response.choices[0].message.content
                _save_cache(single_key, result)
                outputs.append(result)
            except Exception as e:
                print(f"  Individual call failed ({e}), using empty classification")
                outputs.append('{"dm_type":false,"clinical_task":false,"patient_population":false,"data_modality":false,"clinical_outcome":false}')

    return outputs
