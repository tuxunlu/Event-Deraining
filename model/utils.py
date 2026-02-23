from time import perf_counter
from typing import Dict, Tuple

import torch
import torch.nn as nn


def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Return basic parameter statistics for a PyTorch model.

    Returns:
        A dictionary with:
            - total: total number of parameters
            - trainable: number of trainable parameters
            - non_trainable: number of frozen parameters
    """
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    non_trainable = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def format_num_parameters(num_params: int) -> str:
    """
    Format a parameter count in a readable unit.
    """
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.3f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.3f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.3f}K"
    return str(num_params)


def estimate_model_gflops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Estimate FLOPs for one forward pass using torch.profiler.

    Args:
        model: PyTorch model.
        input_shape: Input tensor shape, e.g. (1, 1, 256, 256).
        device: "cpu" or "cuda". If "cuda" is requested but unavailable,
            this function falls back to CPU.

    Returns:
        A dictionary with:
            - flops: total floating-point ops for one forward pass
            - gflops: flops / 1e9
    """
    target_device = device
    if target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"

    model = model.to(target_device).eval()
    sample = torch.randn(*input_shape, device=target_device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if target_device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.no_grad():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = model(sample)

    total_flops = 0
    for event in prof.key_averages():
        event_flops = getattr(event, "flops", 0)
        if event_flops is not None:
            total_flops += int(event_flops)

    return {
        "flops": float(total_flops),
        "gflops": float(total_flops) / 1e9,
    }


def benchmark_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> Dict[str, float]:
    """
    Benchmark forward inference latency.

    Returns timing statistics in milliseconds and FPS.
    """
    target_device = device
    if target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"

    model = model.to(target_device).eval()
    sample = torch.randn(*input_shape, device=target_device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample)
        if target_device == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(timed_runs):
            start = perf_counter()
            _ = model(sample)
            if target_device == "cuda":
                torch.cuda.synchronize()
            end = perf_counter()
            times_ms.append((end - start) * 1000.0)

    times_tensor = torch.tensor(times_ms, dtype=torch.float32)
    mean_ms = float(times_tensor.mean().item())

    return {
        "mean_ms": mean_ms,
        "std_ms": float(times_tensor.std(unbiased=False).item()),
        "min_ms": float(times_tensor.min().item()),
        "max_ms": float(times_tensor.max().item()),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }

if __name__ == "__main__":
    import os
    import sys

    # Allow running this file directly from repo root:
    #   python model/utils.py
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from model.DynamicFourierFilterNet import DynamicFourierFilterNet
        from model.FourierMamba2D import FourierMamba2D
    except ModuleNotFoundError:
        # Fallback for environments that do not resolve the package import.
        from DynamicFourierFilterNet import DynamicFourierFilterNet
        from FourierMamba2D import FourierMamba2D

    run_device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DynamicFourierFilterNet(in_chans=1, out_chans=1, dim=32, num_blocks=4)
    stats = count_model_parameters(model)
    flops = estimate_model_gflops(model, input_shape=(1, 1, 256, 256), device=run_device)
    latency = benchmark_inference_time(
        model,
        input_shape=(1, 1, 256, 256),
        device=run_device,
        warmup_runs=20,
        timed_runs=100,
    )
    print("DFFNet Parameters:")
    print("Total:", format_num_parameters(stats["total"]))
    print("Trainable:", format_num_parameters(stats["trainable"]))
    print("Non-trainable:", format_num_parameters(stats["non_trainable"]))
    print(f"GFLOPs (1x1x256x256): {flops['gflops']:.4f}")
    print(f"Inference ({run_device}) mean latency: {latency['mean_ms']:.3f} ms")
    print(f"Inference ({run_device}) std latency:  {latency['std_ms']:.3f} ms")
    print(f"Inference ({run_device}) throughput:   {latency['fps']:.2f} FPS")

    model = FourierMamba2D(in_chans=1, out_chans=1, dim=32)
    stats = count_model_parameters(model)
    flops = estimate_model_gflops(model, input_shape=(1, 1, 256, 256), device=run_device)
    latency = benchmark_inference_time(
        model,
        input_shape=(1, 1, 256, 256),
        device=run_device,
        warmup_runs=20,
        timed_runs=100,
    )
    print("FourierMamba2D Parameters:")
    print("Total:", format_num_parameters(stats["total"]))
    print("Trainable:", format_num_parameters(stats["trainable"]))
    print("Non-trainable:", format_num_parameters(stats["non_trainable"]))
    print(f"GFLOPs (1x1x256x256): {flops['gflops']:.4f}")
    print(f"Inference ({run_device}) mean latency: {latency['mean_ms']:.3f} ms")
    print(f"Inference ({run_device}) std latency:  {latency['std_ms']:.3f} ms")
    print(f"Inference ({run_device}) throughput:   {latency['fps']:.2f} FPS")
