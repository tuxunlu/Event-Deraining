from time import perf_counter
from typing import Dict, List, Tuple

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

    Returns timing statistics in milliseconds, seconds, and FPS.
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
        "mean_s": mean_ms / 1000.0,
        "std_ms": float(times_tensor.std(unbiased=False).item()),
        "min_ms": float(times_tensor.min().item()),
        "max_ms": float(times_tensor.max().item()),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }


def _format_table(rows: List[Dict[str, str]]) -> str:
    headers = ["Model", "Parameters", "Inference(s)", "GFLOPs"]
    widths = [len(header) for header in headers]
    for row in rows:
        widths[0] = max(widths[0], len(row["Model"]))
        widths[1] = max(widths[1], len(row["Parameters"]))
        widths[2] = max(widths[2], len(row["Inference(s)"]))
        widths[3] = max(widths[3], len(row["GFLOPs"]))

    def fmt_line(values):
        return (
            f"| {values[0]:<{widths[0]}} "
            f"| {values[1]:>{widths[1]}} "
            f"| {values[2]:>{widths[2]}} "
            f"| {values[3]:>{widths[3]}} |"
        )

    sep = (
        f"|{'-' * (widths[0] + 2)}"
        f"|{'-' * (widths[1] + 2)}"
        f"|{'-' * (widths[2] + 2)}"
        f"|{'-' * (widths[3] + 2)}|"
    )

    lines = [fmt_line(headers), sep]
    for row in rows:
        lines.append(
            fmt_line(
                [row["Model"], row["Parameters"], row["Inference(s)"], row["GFLOPs"]]
            )
        )
    return "\n".join(lines)


def benchmark_models_table(
    model_defs: List[Tuple[str, nn.Module]],
    input_shape: Tuple[int, ...],
    device: str,
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> str:
    rows: List[Dict[str, str]] = []
    for model_name, model in model_defs:
        stats = count_model_parameters(model)
        flops = estimate_model_gflops(model, input_shape=input_shape, device=device)
        latency = benchmark_inference_time(
            model,
            input_shape=input_shape,
            device=device,
            warmup_runs=warmup_runs,
            timed_runs=timed_runs,
        )
        rows.append(
            {
                "Model": model_name,
                "Parameters": format_num_parameters(stats["total"]),
                "Inference(s)": f"{latency['mean_s']:.6f}",
                "GFLOPs": f"{flops['gflops']:.4f}",
            }
        )
    return _format_table(rows)

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
        from model.DynamicFourierFilterNetSEKG import DynamicFourierFilterNetSEKG
        from model.DynamicFourierFilterNetFDConv import DynamicFourierFilterNetFDConv
        from model.FourierMamba2D import FourierMamba2D
    except ModuleNotFoundError:
        # Fallback for environments that do not resolve the package import.
        from DynamicFourierFilterNet import DynamicFourierFilterNet
        from DynamicFourierFilterNetSEKG import DynamicFourierFilterNetSEKG
        from DynamicFourierFilterNetFDConv import DynamicFourierFilterNetFDConv
        from FourierMamba2D import FourierMamba2D

    run_device = "cuda" if torch.cuda.is_available() else "cpu"
    input_shape = (1, 1, 256, 256)
    models_to_benchmark: List[Tuple[str, nn.Module]] = [
        ("DynamicFourierFilterNet", DynamicFourierFilterNet(in_chans=1, out_chans=1, dim=32, num_blocks=4)),
        ("DynamicFourierFilterNetSEKG", DynamicFourierFilterNetSEKG(in_chans=1, out_chans=1, dim=32, num_blocks=4)),
        ("DynamicFourierFilterNetFDConv", DynamicFourierFilterNetFDConv(in_chans=1, out_chans=1, dim=32, num_blocks=4)),
        ("FourierMamba2D", FourierMamba2D(in_chans=1, out_chans=1, dim=32, num_blocks=[2, 2, 2, 2])),
    ]

    print(f"Benchmark device: {run_device}")
    print(f"Input shape: {input_shape}")
    print(
        benchmark_models_table(
            model_defs=models_to_benchmark,
            input_shape=input_shape,
            device=run_device,
            warmup_runs=20,
            timed_runs=500,
        )
    )
