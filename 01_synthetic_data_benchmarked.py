import argparse
import asyncio
import json
import math
from pathlib import Path
import time
import aiohttp
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# --- Hardware Benchmarking Tracker ---
class BenchmarkMetrics:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.total_eval_tokens = 0
        self.total_prompt_tokens = 0
        self.total_eval_duration_ns = 0
        self.completed_requests = 0

    async def log_request(self, res_json):
        async with self.lock:
            self.completed_requests += 1
            # Ollama returns timing metrics in nanoseconds
            self.total_eval_tokens += res_json.get("eval_count", 0)
            self.total_prompt_tokens += res_json.get("prompt_eval_count", 0)
            self.total_eval_duration_ns += res_json.get("eval_duration", 0)

    def report(self, elapsed_wall_time):
        total_eval_sec = self.total_eval_duration_ns / 1e9
        gen_tps = self.total_eval_tokens / total_eval_sec if total_eval_sec > 0 else 0
        system_tps = self.total_eval_tokens / elapsed_wall_time if elapsed_wall_time > 0 else 0

        print("\n" + "="*50)
        print("          HARDWARE BENCHMARK SUMMARY          ")
        print("="*50)
        print(f"Total Requests Completed : {self.completed_requests}")
        print(f"Total Generated Tokens  : {self.total_eval_tokens}")
        print(f"Total Prompt Tokens     : {self.total_prompt_tokens}")
        print(f"Wall Clock Execution    : {elapsed_wall_time:.2f} s")
        print(f"Generation Speed (GPU)  : {gen_tps:.2f} tokens/sec")
        print(f"System Throughput (Wall): {system_tps:.2f} tokens/sec")
        print("="*50 + "\n")

metrics = BenchmarkMetrics()


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Ollama Throughput Benchmark")
    parser.add_argument("input_file", type=Path, help="Path to input CSV")
    parser.add_argument("--concurrency", type=int, default=10, help="Max parallel API calls")
    parser.add_argument("--model", type=str, default="gemma3:4b", help="Ollama model tag")
    return parser.parse_args()


# --- Async Ollama Generation ---
async def generate_single_ollama(session, semaphore, prompt, model="gemma3:4b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Return aggregated JSON with server-side metrics
    }

    async with semaphore:
        try:
            async with session.post(url, json=payload) as r:
                res = await r.json()
                await metrics.log_request(res)
                return res.get("response", "")
        except Exception as e:
            print(f"Request failed: {e}")
            return ""


async def generate_text_ollama_batched(session, semaphore, prompt, model="gemma3:4b", num_return_sequences=3):
    """Fires requests concurrently up to the semaphore concurrency limit."""
    tasks = [
        generate_single_ollama(session, semaphore, prompt, model)
        for _ in range(num_return_sequences)
    ]
    return await asyncio.gather(*tasks)


# --- Fitness Calculation ---
def fitness_score(original, candidate):
    if len(candidate.strip()) == 0 or candidate.strip() == ".":
        return 0.0

    embeddings = sbert.encode([original, candidate], convert_to_tensor=True)
    sim = util.cos_sim(embeddings[0], embeddings[1]).item()

    orig_words = set(original.lower().split())
    cand_words = set(candidate.lower().split())
    overlap_ratio = len(orig_words & cand_words) / len(cand_words) if cand_words else 1.0

    diversity = 1.0 - overlap_ratio
    return 2 * sim + diversity


# --- Async Evolutionary Corpus Generation ---
async def evolve_corpus_async(
    session,
    semaphore,
    text,
    target_size=10,
    offspring_per_sample=5,
    retain_top=10,
    generations=3,
    model="gemma3:4b",
    prompt="Paraphrase this:"
):
    population = [text]
    all_candidates = []

    for gen in range(generations):
        # Dispatch generation for all parents concurrently
        generation_tasks = [
            generate_text_ollama_batched(
                session, semaphore, f"{prompt} {parent}", model, offspring_per_sample
            )
            for parent in population
        ]
        results = await asyncio.gather(*generation_tasks)

        gen_candidates = []
        for parent, candidates in zip(population, results):
            for c in candidates:
                score = fitness_score(text, c)
                # Store candidate along with metadata; preserving original root text reference
                gen_candidates.append((c, score, gen + 1, text, parent))

        all_candidates.extend(gen_candidates)
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        survivors = all_candidates[:retain_top]
        population = [c[0] for c in survivors]

    all_candidates.sort(key=lambda x: x[1], reverse=True)
    return all_candidates[:target_size]


# --- Main Benchmarking Loop ---
async def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)
    
    # Adjusted sample size for realistic benchmarking runs
    seed_corpus = df.sample(min(1000, len(df)), random_state=330)["all_text"].tolist()

    semaphore = asyncio.Semaphore(args.concurrency)
    prompt_prefix = (
        "Paraphrase this text. Do not include options or extra text. "
        "The text is: "
    )

    start_wall_time = time.perf_counter()

    # Reuse TCP connection pool via aiohttp ClientSession
    async with aiohttp.ClientSession() as session:
        with open("text_outputs_ollama.json", "a") as out_file:
            for root_text in tqdm(seed_corpus, desc="Benchmarking Corpus"):
                synthetic_text = await evolve_corpus_async(
                    session=session,
                    semaphore=semaphore,
                    text=root_text,
                    target_size=10,
                    offspring_per_sample=5,
                    retain_top=10,
                    generations=3,  # Adjusted to 3 generations for benchmarking
                    model=args.model,
                    prompt=prompt_prefix,
                )

                # Persist output safely
                for syn, score, gen, orig_text, parent in synthetic_text:
                    record = {
                        "syn": syn,
                        "score": score,
                        "gen": gen,
                        "orig_text": orig_text,
                        "parent": parent,
                        "model": args.model,
                    }
                    out_file.write(json.dumps(record) + "\n")

    elapsed_wall_time = time.perf_counter() - start_wall_time
    metrics.report(elapsed_wall_time)


if __name__ == "__main__":
    asyncio.run(main())
