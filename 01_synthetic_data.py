import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


device = "cuda" if torch.cuda.is_available() else "cpu"
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def calculate_perplexity(text):
    """Return perplexity of a given text under the chosen T5/FLAN model.
    Probably want to switch this to GPT since we would actually get
    log probs from that
    
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss
    return math.exp(loss.item())


def truncate_to_sentence(text):
    """
    option to truncate the text at an end of sentence publication
    so the model doesn't ramble on
    """
    for end in [".", "!", "?"]:
        if end in text:
            return text[: text.index(end)+1]
    return text


def generate_text_ollama(prompt, model='gemma3:4b', num_return_sequences=3):
    """
    Generate text with Ollama, which is a local LLM.
    The prompt is passed directly, output is processed from the Ollama response. 
    """
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "gemma3:4b",
        "prompt": prompt
    }
    
    generated_texts = []
    
    for i in range(num_return_sequences):
        r = requests.post(url, data=json.dumps(data))
        text = [json.loads(s) for s in r.text.split('\n') if s]
        synth_text = ''.join([t['response'] for t in text])
        generated_texts.append(synth_text)

    return generated_texts


def fitness_score(original, candidate, verbose=False):
    """Evaluate fitness of a candidate text (semantic similarity + low word overlap + fluency)."""
    if len(candidate.strip()) == 0 or candidate.strip() == '.':
        return 0
    
    # get SBERT embeddings
    embeddings = sbert.encode([original, candidate], convert_to_tensor=True)

    # semantic similarity
    sim = util.cos_sim(embeddings[0], embeddings[1]).item()

    # word overlap with original
    orig_words = set(original.lower().split())
    cand_words = set(candidate.lower().split())
    if len(cand_words) == 0:
        overlap_ratio = 1.0  # penalize empty candidate
    else:
        overlap_ratio = len(orig_words & cand_words) / len(cand_words)

    # define diversity as inverse of overlap ratio
    diversity = 1 - overlap_ratio  # higher = more diverse wording

    
    return 2 * sim + diversity


def evolve_corpus(text, target_size=100, offspring_per_sample=5, retain_top=20, generations=3,
                  verbose=False, prompt="Paraphrase this:"):
    """
    Expand a corpus using a global survivor-pool GA.
    - Tracks which generation each candidate came from.
    - Returns the final top `target_size` samples with metadata.
    """
    population = [text]  # current parents
    all_candidates = []     # store candidates across gens

    for gen in range(generations):
        gen_candidates = []

        # 1. Generate offspring for each parent
        for parent in population:
            candidates = generate_text_ollama(
                f"{prompt} {parent}",
                num_return_sequences=offspring_per_sample
            )
            # score each candidate against its parent
            scored = [(c, fitness_score(text, c), gen+1, text) for c in candidates]
            gen_candidates.extend(scored)
            
            for item in scored:
                candidate, score, generation, text = item
                to_output = {
                    'candidate':candidate,
                    'score':score,
                    'generation':generation,
                    'text':text,
                    'prompt':prompt,
                    'parent':parent
                }
                with open('candidate_texts_ollama.json', 'a+') as op:
                    op.write(json.dumps(to_output) + '\n')

        # 2. Add this generation’s candidates to the archive
        all_candidates.extend(gen_candidates)
        
        # 3. Sort within this generation and keep survivors
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        survivors = all_candidates[:retain_top]

        # 4. Survivors become next gen parents
        population = [c for c, _, _, _ in survivors]

        if verbose:
            print(f"Generation {gen+1}: {len(gen_candidates)} candidates, kept {len(population)} survivors")

    # finally, select global top n 
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    final_population = all_candidates[:target_size]

    return final_population

if __name__ == '__main__':
    all_synth_texts = []

    seed_corpus = ['This is my first sentence', 'This is my second sentence']
    MODEL_NAME = 'gemma3:4b'

    for text in tqdm(seed_corpus):
        synthetic_text = evolve_corpus(
            text,
            target_size=10,
            offspring_per_sample=5,
            retain_top=10,
            generations=20,
            prompt=("Paraphrase this text. Do not include options or any other text around it. "
                    "Simply include the new text with no other caveats, examples, or options. "
                    "The text is: "
                )
        )

        for new_sample in synthetic_text:
            syn, score, gen, text = new_sample
            to_output = {'syn':syn, 'score':score, 'gen':gen, 'text':text}
            to_output['model'] = MODEL_NAME
            to_output['target_size'] = 10
            to_output['offspring_per_sample'] = 5
            to_output['retain_top'] = 10
            to_output['generations'] = 20
            
            with open('text_outputs_ollama.json', 'a+') as op:
                op.write(json.dumps(to_output) + '\n')
                
        all_synth_texts.extend(synthetic_text)