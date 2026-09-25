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

# Enable CPU multithreading optimizations
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process an input file for BERTopic inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python predict.py input.txt --limit 1000 --model bertopic_model.pkl"
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input file"
    )

    parser.add_argument(
        "-m", "--model",
        type=Path,
        default=Path("bertopic_model.pkl"),
        help="Path to the trained BERTopic joblib model file (default: bertopic_model.pkl)"
    )

    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process for quick benchmarking (e.g., 500)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation on CPU (default: 64)"
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
    print(f"Loaded {num_docs:,} documents for inference in {t1_data - t0_data:.2f} seconds.")

    # --- Step 2: Load Saved Model ---
    t0_model = time.perf_counter()
    print(f"Loading trained model from {args.model}...")
    topic_model = joblib.load(args.model)
    t1_model = time.perf_counter()
    print(f"Model loaded in {t1_model - t0_model:.2f} seconds.")

    # --- Step 3: Fast Parallelized Embedding Generation ---
    print("Generating embeddings across available CPU cores...")
    t0_embed = time.perf_counter()

    # Extract underlying SentenceTransformer instance from BERTopic wrapper wrapper
    if hasattr(topic_model.embedding_model, "embedding_model"):
        st_model = topic_model.embedding_model.embedding_model
    else:
        st_model = topic_model.embedding_model

    # Run multi-process encoding on the raw SentenceTransformer object
    pool = st_model.start_multi_process_pool()
    embeddings = st_model.encode_multi_process(
        documents, 
        pool, 
        batch_size=args.batch_size
    )
    st_model.stop_multi_process_pool(pool)
    
    t1_embed = time.perf_counter()
    embed_time = t1_embed - t0_embed
    embed_throughput = num_docs / embed_time if embed_time > 0 else 0

    # --- Step 4: Perform Topic Inference/Transform ---
    print("Assigning topics to new documents...")
    t0_predict = time.perf_counter()
    
    # Pass pre-computed embeddings directly into transform()
    topics, probs = topic_model.transform(documents, embeddings=embeddings)
    
    t1_predict = time.perf_counter()
    predict_time = t1_predict - t0_predict
    predict_throughput = num_docs / predict_time if predict_time > 0 else 0

    total_inference_time = embed_time + predict_time
    total_throughput = num_docs / total_inference_time if total_inference_time > 0 else 0

    # --- Benchmarking Metrics Output ---
    print("\n" + "="*45)
    print("         INFERENCE BENCHMARK RESULTS       ")
    print("="*45)
    print(f"Total Documents Inferred  : {num_docs:,}")
    print(f"CPU Cores Utilized        : {os.cpu_count()}")
    print("-" * 45)
    print(f"Embedding Generation Time : {embed_time:.2f} s")
    print(f"Embedding Throughput      : {embed_throughput:.2f} docs/sec")
    print("-" * 45)
    print(f"Transform/Prediction Time : {predict_time:.2f} s")
    print(f"Transform Throughput      : {predict_throughput:.2f} docs/sec")
    print("-" * 45)
    print(f"Total Prediction Time     : {total_inference_time:.2f} s")
    print(f"OVERALL INFERENCE SPEED   : {total_throughput:.2f} docs/sec")
    print("="*45)
