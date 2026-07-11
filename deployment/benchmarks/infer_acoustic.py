"""
ONNX Runtime inference benchmark for DiffSinger acoustic models.

Supports precision-aware benchmarking across fp32, fp16, bf16, and int8 models.

Usage::

    python infer_acoustic.py --model model_fp16.onnx --precision fp16 --n_runs 100
    python infer_acoustic.py --model model.onnx --precision all          # compare all
    python infer_acoustic.py --model model.onnx --model2 model_int8.onnx # A/B compare
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_N_TOKENS = 10
DEFAULT_N_FRAMES = 100
DEFAULT_N_RUNS = 100
DEFAULT_WARMUP = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_providers(precision: str, device: str = 'cuda') -> list:
    """Return sensible ONNX Runtime execution providers."""
    if device == 'cpu':
        return ['CPUExecutionProvider']

    provider_map = {
        'fp32':  ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'fp16':  ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'],
        'bf16':  ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        'int8':  ['TensorrtExecutionProvider', 'CPUExecutionProvider'],
    }
    return provider_map.get(precision, ['CPUExecutionProvider'])


def create_dummy_inputs(n_tokens=DEFAULT_N_TOKENS, n_frames=DEFAULT_N_FRAMES,
                        input_names=None):
    """Generate dummy inputs matching the model's expected input signature."""
    feed = {
        'tokens':    np.array([[1] * n_tokens], dtype=np.int64),
        'durations': np.array([[n_frames // n_tokens] * n_tokens], dtype=np.int64),
        'f0':        np.array([[440.] * n_frames], dtype=np.float32),
    }
    # Add common optional inputs
    if input_names:
        for name in input_names:
            if name in ('tokens', 'durations', 'f0'):
                continue
            if name == 'speedup':
                feed[name] = np.array([20], dtype=np.int64)
            elif name == 'gender':
                feed[name] = np.array([[0.] * n_frames], dtype=np.float32)
            elif name == 'velocity':
                feed[name] = np.array([[1.] * n_frames], dtype=np.float32)
            elif name == 'spk_embed':
                feed[name] = np.random.randn(1, n_frames, 256).astype(np.float32)
            elif name == 'languages':
                feed[name] = np.zeros((1, n_tokens), dtype=np.int64)
            else:
                feed[name] = np.zeros((1, n_frames), dtype=np.float32)
    return feed


def get_input_names(session: ort.InferenceSession) -> list:
    return [inp.name for inp in session.get_inputs()]


def get_output_names(session: ort.InferenceSession) -> list:
    return [out.name for out in session.get_outputs()]


def benchmark(session, feed, n_runs=DEFAULT_N_RUNS, warmup=DEFAULT_WARMUP):
    """Run the benchmark and return stats."""
    # Warmup
    for _ in range(warmup):
        session.run(None, feed)

    times = []
    for _ in tqdm.tqdm(range(n_runs), desc='Benchmarking'):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000),
        'p50_ms': float(np.percentile(times, 50) * 1000),
        'p95_ms': float(np.percentile(times, 95) * 1000),
        'n_runs': n_runs,
    }


def print_benchmark(name: str, stats: dict):
    print(f'\n=== {name} ===')
    print(f'  mean:  {stats["mean_ms"]:.2f} ms')
    print(f'  std:   {stats["std_ms"]:.2f} ms')
    print(f'  p50:   {stats["p50_ms"]:.2f} ms')
    print(f'  p95:   {stats["p95_ms"]:.2f} ms')
    print(f'  min:   {stats["min_ms"]:.2f} ms')
    print(f'  max:   {stats["max_ms"]:.2f} ms')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ONNX Acoustic Model Benchmark')
    parser.add_argument('--model', type=Path, required=True,
                        help='Path to ONNX model file.')
    parser.add_argument('--model2', type=Path, default=None,
                        help='Optional second model for A/B comparison.')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16', 'bf16', 'int8', 'all'],
                        help='Precision hint for selecting execution providers.')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on.')
    parser.add_argument('--n_tokens', type=int, default=DEFAULT_N_TOKENS)
    parser.add_argument('--n_frames', type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument('--n_runs', type=int, default=DEFAULT_N_RUNS)
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP)

    args = parser.parse_args()

    if not args.model.exists():
        print(f'Error: model not found: {args.model}')
        sys.exit(1)

    # Run benchmark for primary model
    print(f'Loading model: {args.model} (precision={args.precision}, device={args.device})')
    providers = resolve_providers(args.precision, args.device)

    session = ort.InferenceSession(
        str(args.model),
        providers=providers,
        sess_options=_make_session_options(),
    )
    input_names = get_input_names(session)
    print(f'  Inputs: {input_names}')
    print(f'  Outputs: {get_output_names(session)}')
    print(f'  Providers: {session.get_providers()}')

    feed = create_dummy_inputs(args.n_tokens, args.n_frames, input_names=input_names)
    stats = benchmark(session, feed, args.n_runs, args.warmup)
    print_benchmark(f'Model: {args.model.name}', stats)

    # A/B comparison
    if args.model2 and args.model2.exists():
        print(f'\nLoading model2: {args.model2}')
        session2 = ort.InferenceSession(str(args.model2), providers=providers,
                                        sess_options=_make_session_options())
        feed2 = create_dummy_inputs(args.n_tokens, args.n_frames, input_names=get_input_names(session2))
        stats2 = benchmark(session2, feed2, args.n_runs, args.warmup)
        print_benchmark(f'Model: {args.model2.name}', stats2)

        speedup = stats['mean_ms'] / stats2['mean_ms'] if stats2['mean_ms'] > 0 else 0
        print(f'\n=== Speedup (model1 / model2) ===')
        print(f'  {speedup:.2f}x')


def _make_session_options():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    return opts


if __name__ == '__main__':
    main()

