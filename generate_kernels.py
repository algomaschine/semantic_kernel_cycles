#!/usr/bin/env python3
"""
Generate archetype_kernels.json using Ollama with a lightweight model.
- Uses qwen2.5:1.5b (or phi3:mini if available)
- Validates each word is a real English noun (WordNet)
- No duplicate roots within an archetype
- No shared words across archetypes
- No placeholders – if not enough real words, reduces target count dynamically
"""

import json
import requests
import time
import sys
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import SnowballStemmer

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ======================= CONFIGURATION =======================
ARCHETYPES = [
    'innocent', 'orphan', 'warrior', 'caregiver', 'seeker', 'lover',
    'destroyer', 'creator', 'ruler', 'magician', 'sage', 'jester'
]
MODEL_NAME = "qwen2.5:1.5b"   # lightweight, works on CPU
OLLAMA_URL = "http://localhost:11434"
TARGET_WORDS = 8              # total words per archetype (including archetype itself)
MIN_ADDITIONAL = 5            # minimum additional real words required
MAX_RETRIES = 4
BASE_TEMPERATURE = 0.3
TEMPERATURE_STEP = 0.1
# Quality thresholds (0‑10) – relaxed because small model
MIN_RELEVANCE = 4
MIN_SEARCHABILITY = 3
REQUIRE_UNAMBIGUOUS = False

# Global set to track all used words across archetypes
USED_WORDS_GLOBAL = set()
stemmer = SnowballStemmer("english")

def ensure_model():
    """Check if model exists, if not pull it (using subprocess)."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        if not any(m.get("name", "").startswith(MODEL_NAME) for m in models):
            print(f"Model {MODEL_NAME} not found. Pulling (this may take a few minutes)...")
            import subprocess
            subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
    except Exception as e:
        print(f"Warning: Could not verify model: {e}")

def is_noun(word):
    """Check if a word is a noun using WordNet."""
    synsets = wn.synsets(word)
    for s in synsets:
        if s.pos() == 'n':
            return True
    return False

def query_ollama(prompt, temperature, max_tokens=1200):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    }
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"  Error: {e}")
        return ""

def build_prompt(archetype, additional_needed, used_words_list):
    used_str = ", ".join(used_words_list) if used_words_list else "none"
    return f"""Generate a JSON array of exactly {additional_needed} objects for the Jungian archetype "{archetype}".

Each object must have fields:
- "word": a single, real, common English noun (lowercase, no spaces). Must be a noun (e.g., "courage", "battle").
- "relevance": integer 0-10 (how central to the archetype).
- "searchability": integer 0-10 (how likely as a Google search keyword).
- "unambiguous": boolean (true if single strong meaning).

Rules:
- All words must be REAL ENGLISH NOUNS (not adjectives/verbs). No made‑up words.
- No word may share the same root with another word in the same archetype.
- The following words are already used in other archetypes and MUST NOT appear: {used_str}. Choose completely different words.

Return ONLY the JSON array. Example:
[
  {{"word": "battle", "relevance": 9, "searchability": 8, "unambiguous": true}},
  {{"word": "courage", "relevance": 9, "searchability": 7, "unambiguous": true}}
]

Your response:"""

def parse_candidate_words(response, archetype, existing_archetype_words):
    """Extract JSON, apply filters, return list of valid new words."""
    response = re.sub(r'^```json\s*', '', response.strip())
    response = re.sub(r'^```\s*', '', response)
    response = re.sub(r'\s*```$', '', response)
    try:
        data = json.loads(response)
        if not isinstance(data, list):
            raise ValueError
    except:
        match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except:
                data = []
        else:
            data = []
    valid = []
    archetype_root = stemmer.stem(archetype)
    for item in data:
        if not isinstance(item, dict):
            continue
        word = str(item.get("word", "")).strip().lower()
        relevance = item.get("relevance", 0)
        searchability = item.get("searchability", 0)
        unambiguous = item.get("unambiguous", False)
        # Basic filters
        if ' ' in word or not word.isalpha() or len(word) < 3:
            continue
        if not is_noun(word):
            continue
        if word in USED_WORDS_GLOBAL:
            continue
        if word in existing_archetype_words:
            continue
        # Root uniqueness
        word_root = stemmer.stem(word)
        if word_root == archetype_root:
            continue
        if any(stemmer.stem(w) == word_root for w in existing_archetype_words):
            continue
        if (relevance >= MIN_RELEVANCE and
            searchability >= MIN_SEARCHABILITY and
            (not REQUIRE_UNAMBIGUOUS or unambiguous)):
            valid.append(word)
    # Remove duplicates
    return list(dict.fromkeys(valid))

def expand_archetype(archetype):
    """Generate words for one archetype."""
    global USED_WORDS_GLOBAL
    additional_needed = TARGET_WORDS - 1
    existing_archetype_words = []
    for attempt in range(MAX_RETRIES):
        temp = BASE_TEMPERATURE + (attempt * TEMPERATURE_STEP)
        print(f"    Attempt {attempt+1}, temperature={temp:.2f}")
        used_list = list(USED_WORDS_GLOBAL)
        prompt = build_prompt(archetype, additional_needed, used_list)
        response = query_ollama(prompt, temperature=temp)
        if not response:
            continue
        candidates = parse_candidate_words(response, archetype, existing_archetype_words)
        print(f"      Valid new words: {len(candidates)}")
        for w in candidates:
            if len(existing_archetype_words) >= additional_needed:
                break
            existing_archetype_words.append(w)
            USED_WORDS_GLOBAL.add(w)
        if len(existing_archetype_words) >= additional_needed:
            break
        time.sleep(2)
    # Build final list
    result = [archetype] + existing_archetype_words[:additional_needed]
    # If we have fewer than TARGET_WORDS, we'll still return what we have (no placeholders)
    print(f"    Final: {len(result)} words (target {TARGET_WORDS})")
    return result

def main():
    print("Ensuring LLM model is available...")
    ensure_model()
    print(f"Using Ollama model: {MODEL_NAME}\n")

    kernels = {}
    global USED_WORDS_GLOBAL
    USED_WORDS_GLOBAL.clear()
    for arch in ARCHETYPES:
        print(f"\nExpanding '{arch}'...")
        result = expand_archetype(arch)
        kernels[arch] = result
        print(f"  -> {len(result)} words: {result[:5]}{'...' if len(result)>5 else ''}")
        time.sleep(1)

    with open('archetype_kernels.json', 'w', encoding='utf-8') as f:
        json.dump(kernels, f, indent=2, ensure_ascii=False)

    print("\n✅ Saved to archetype_kernels.json")
    print(f"Total unique words across all archetypes: {len(USED_WORDS_GLOBAL)}")

if __name__ == "__main__":
    main()
