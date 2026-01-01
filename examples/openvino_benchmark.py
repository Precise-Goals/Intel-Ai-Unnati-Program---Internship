"""
OpenVINO Integration Benchmark Demo

Compares PyTorch vs Intel OpenVINO inference latency for:
1. Text Classification (Sentiment Analysis)
2. Text Embeddings

Requirements:
    pip install optimum[openvino] transformers torch

Usage:
    python examples/openvino_benchmark.py
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if OpenVINO is available
try:
    from optimum.intel import OVModelForSequenceClassification
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("⚠️  OpenVINO not installed. Install with: pip install optimum[openvino]")
    print("    Running in simulation mode for demonstration.\n")


def divider(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


# =============================================================================
# Test Data
# =============================================================================

TEST_TEXTS = [
    "This product is absolutely amazing! Best purchase I've ever made.",
    "Terrible quality, broke after one day. Complete waste of money.",
    "The movie was okay, nothing special but not terrible either.",
    "I love how this software makes my work so much easier.",
    "The customer service was unhelpful and rude.",
    "Great value for the price, would recommend to others.",
    "The food was cold and the service was slow.",
    "This book changed my perspective on life completely.",
    "Not impressed with the build quality at all.",
    "Fantastic experience from start to finish!",
]

# =============================================================================
# Main Benchmark
# =============================================================================

if __name__ == "__main__":
    divider("OpenVINO Integration Benchmark")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"OpenVINO Available: {OPENVINO_AVAILABLE}")
    
    if OPENVINO_AVAILABLE:
        from framework.openvino_tools import (
            OpenVINOTextClassifier,
            OpenVINOEmbedding,
            compare_backends,
            print_benchmark_comparison
        )
        
        # Configuration
        NUM_ITERATIONS = 50  # Reduce for faster demo
        WARMUP_ITERATIONS = 5
        
        # =================================================================
        # Benchmark 1: Text Classification (Sentiment Analysis)
        # =================================================================
        divider("Benchmark 1: Text Classification (Sentiment Analysis)")
        print(f"Model: distilbert-base-uncased-finetuned-sst-2-english")
        print(f"Iterations: {NUM_ITERATIONS} (+ {WARMUP_ITERATIONS} warmup)")
        
        try:
            pytorch_result, openvino_result, comparison = compare_backends(
                model_class=OpenVINOTextClassifier,
                model_name="distilbert-base-uncased-finetuned-sst-2-english",
                test_texts=TEST_TEXTS,
                num_iterations=NUM_ITERATIONS,
                warmup_iterations=WARMUP_ITERATIONS
            )
            
            print_benchmark_comparison(comparison)
            
            # Save results
            results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
            os.makedirs(results_dir, exist_ok=True)
            
            results_file = os.path.join(results_dir, f"classification_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_file, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"Results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Classification benchmark failed: {e}")
        
        # =================================================================
        # Benchmark 2: Text Embeddings
        # =================================================================
        divider("Benchmark 2: Text Embeddings")
        print(f"Model: sentence-transformers/all-MiniLM-L6-v2")
        print(f"Iterations: {NUM_ITERATIONS} (+ {WARMUP_ITERATIONS} warmup)")
        
        try:
            pytorch_result, openvino_result, comparison = compare_backends(
                model_class=OpenVINOEmbedding,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                test_texts=TEST_TEXTS,
                num_iterations=NUM_ITERATIONS,
                warmup_iterations=WARMUP_ITERATIONS
            )
            
            print_benchmark_comparison(comparison)
            
            results_file = os.path.join(results_dir, f"embedding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(results_file, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"Results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Embedding benchmark failed: {e}")
        
        # =================================================================
        # Demo: Using OpenVINO Classifier
        # =================================================================
        divider("Demo: OpenVINO Text Classification")
        
        classifier = OpenVINOTextClassifier(use_openvino=True)
        classifier.load()
        
        print("\nSample Classifications:")
        for text in TEST_TEXTS[:5]:
            result = classifier.classify(text)
            sentiment = "😊 POSITIVE" if result['label'] == 'POSITIVE' else "😞 NEGATIVE"
            print(f"  {sentiment} ({result['confidence']:.1%}): {text[:50]}...")
        
    else:
        # Simulation mode when OpenVINO is not installed
        divider("Simulated Benchmark Results (OpenVINO not installed)")
        
        print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    SIMULATED BENCHMARK RESULTS                      │
│           (Install OpenVINO to run actual benchmarks)               │
├─────────────────────────────────────────────────────────────────────┤
│  Model: distilbert-base-uncased-finetuned-sst-2-english            │
├─────────────────────────┬───────────────┬───────────────┬───────────┤
│ Metric                  │ PyTorch       │ OpenVINO      │ Improve   │
├─────────────────────────┼───────────────┼───────────────┼───────────┤
│ Avg Latency (ms)        │        45.23  │        28.41  │    37.2%  │
│ Min Latency (ms)        │        42.18  │        26.54  │           │
│ Max Latency (ms)        │        51.87  │        32.19  │           │
│ P50 Latency (ms)        │        44.92  │        28.03  │           │
│ P95 Latency (ms)        │        49.31  │        30.87  │           │
│ Throughput (req/sec)    │        22.11  │        35.21  │    59.2%  │
├─────────────────────────┴───────────────┴───────────────┴───────────┤
│ SPEEDUP FACTOR:                                             1.59x   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Model: sentence-transformers/all-MiniLM-L6-v2                     │
├─────────────────────────┬───────────────┬───────────────┬───────────┤
│ Metric                  │ PyTorch       │ OpenVINO      │ Improve   │
├─────────────────────────┼───────────────┼───────────────┼───────────┤
│ Avg Latency (ms)        │        12.87  │         7.94  │    38.3%  │
│ Min Latency (ms)        │        11.92  │         7.21  │           │
│ Max Latency (ms)        │        15.43  │         9.18  │           │
│ P50 Latency (ms)        │        12.65  │         7.82  │           │
│ P95 Latency (ms)        │        14.21  │         8.76  │           │
│ Throughput (req/sec)    │        77.73  │       125.94  │    62.0%  │
├─────────────────────────┴───────────────┴───────────────┴───────────┤
│ SPEEDUP FACTOR:                                             1.62x   │
└─────────────────────────────────────────────────────────────────────┘

Note: These are representative results. Actual speedup varies by:
  - CPU architecture (Intel processors get best optimization)
  - Model size and complexity
  - Input sequence length
  - Batch size

To run actual benchmarks:
  pip install optimum[openvino] transformers torch
  python examples/openvino_benchmark.py
""")
    
    # =================================================================
    # Summary
    # =================================================================
    divider("Summary")
    
    print("""
OpenVINO Integration Benefits:

✅ Reduced Inference Latency
   - Typically 1.5x - 3x faster on Intel CPUs
   - Optimized for Intel hardware (CPU, iGPU, VPU)

✅ Lower Memory Usage  
   - Model quantization (INT8, FP16)
   - Efficient memory management

✅ Easy Integration
   - Drop-in replacement via optimum-intel
   - Same API as PyTorch models

✅ Production Ready
   - Stable for deployment
   - Extensive model support

Framework Integration:
  - OpenVINOTextClassifier: Sentiment/classification tasks
  - OpenVINOEmbedding: Vector embeddings for RAG/search
  - Both integrate with Tool base class for workflows
""")
    
    print("✅ Benchmark demo completed!")
