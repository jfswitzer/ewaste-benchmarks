'''
MB: This is the sample code from the BERTopic website with some
placeholder code for loading StackOverflow data. This currently
does both fitting and prediction, but we can split that into separate tasks.  
'''
import argparse
import joblib
import os
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import xmltodict

# Enable CPU multithreading optimizations across libraries
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process an input file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python main.py input.txt --verbose"
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input file"
    )
    
    # Benchmarking flags
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation on CPU (default: 64)"
    )

    # Added limit flag (default to None so full dataset runs if omitted)
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process for quick benchmarking (e.g., 500)"
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # --- Step 1: Data Ingestion & Parsing ---
    t0_data = time.perf_counter()
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    xml_dict = xmltodict.parse(data)
    comments_list = xml_dict['comments']['row']
    
    # Handle single element vs list output from xmltodict
    if isinstance(comments_list, dict):
        comments_list = [comments_list]
        
    documents = [c['@Text'] for c in comments_list if '@Text' in c]

    # --- Slice to subset if --limit is specified ---
    if args.limit and args.limit > 0:
        documents = documents[:args.limit]
    
    t1_data = time.perf_counter()
    
    num_docs = len(documents)
    print(f"Loaded {num_docs:,} documents in {t1_data - t0_data:.2f} seconds.")

    # --- Step 2: Initialize Embedding Model ---
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Configure SentenceTransformer to use all CPU threads & optimized PyTorch device
    embedding_model.max_seq_length = 256  # Keeps memory footprint lean and processing fast
    
    # --- Step 3: Explicit CPU Benchmarking of BERTopic ---
    # We pre-compute embeddings using encode_multi_process or batched CPU encoding to capture pure throughput
    print("Generating embeddings across available CPU cores...")
    t0_embed = time.perf_counter()
    
    # Start multi-process pool across CPU cores for SentenceTransformer
    pool = embedding_model.start_multi_process_pool()
    embeddings = embedding_model.encode_multi_process(
        documents, 
        pool, 
        batch_size=args.batch_size
    )
    embedding_model.stop_multi_process_pool(pool)
    
    t1_embed = time.perf_counter()
    embed_time = t1_embed - t0_embed
    embed_throughput = num_docs / embed_time if embed_time > 0 else 0

    # Pass pre-computed embeddings to BERTopic fitting phase
    print("Fitting BERTopic model (Dimensionality Reduction & Clustering)...")
    t0_fit = time.perf_counter()
    
    # BERTopic uses pre-calculated embeddings directly (n_jobs=-1 leverages CPU cores for HDBSCAN/UMAP)
    topic_model = BERTopic(embedding_model=embedding_model, calculate_probabilities=False)
    topics, probs = topic_model.fit_transform(documents, embeddings=embeddings)
    
    t1_fit = time.perf_counter()
    fit_time = t1_fit - t0_fit
    
    total_pipeline_time = (t1_fit - t0_data)
    total_throughput = num_docs / total_pipeline_time if total_pipeline_time > 0 else 0

    # Save output
    joblib.dump(topic_model, "bertopic_model.pkl")

    # --- Benchmarking Metrics Output ---
    print("\n" + "="*45)
    print("           BENCHMARKING RESULTS           ")
    print("="*45)
    print(f"Total Documents Processed : {num_docs:,}")
    print(f"CPU Cores Utilized        : {os.cpu_count()}")
    print("-" * 45)
    print(f"Embedding Generation Time : {embed_time:.2f} s")
    print(f"Embedding Throughput      : {embed_throughput:.2f} docs/sec")
    print("-" * 45)
    print(f"BERTopic Fitting Time     : {fit_time:.2f} s")
    print(f"Total Pipeline Execution  : {total_pipeline_time:.2f} s")
    print(f"OVERALL THROUGHPUT        : {total_throughput:.2f} docs/sec")
    print("="*45)
