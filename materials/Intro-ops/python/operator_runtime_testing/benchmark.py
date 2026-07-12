from __future__ import annotations

from collections.abc import Callable

import torch
from torch.autograd.profiler import DeviceType

_l2_flush_cache: torch.Tensor | None = None


def _get_l2_flush_cache() -> torch.Tensor:
    global _l2_flush_cache
    if _l2_flush_cache is None:
        l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
        if l2_bytes <= 0:
            l2_bytes = int(256e6)
        _l2_flush_cache = torch.empty(max(l2_bytes // 4, 1), dtype=torch.int, device="cuda")
    return _l2_flush_cache


def _sum_cuda_kernel_ms(kineto_results) -> float:
    total_us = 0.0
    for evt in kineto_results.events():
        if evt.device_type() != DeviceType.CUDA:
            continue
        name = evt.name()
        if "vectorized_elementwise" in name and "FillFunctor" in name:
            continue
        total_us += evt.duration_ns() / 1000.0
    return total_us / 1000.0


def _build_arg_pool(
    args: tuple[object, ...],
    *,
    pool_size: int,
    max_clone_bytes: int,
) -> list[tuple[object, ...]] | None:
    if not args:
        return None

    tensor_mask = tuple(isinstance(arg, torch.Tensor) for arg in args)
    total_bytes = sum(
        arg.nelement() * arg.element_size()
        for arg, is_tensor in zip(args, tensor_mask, strict=True)
        if is_tensor
    )
    if total_bytes * pool_size > max_clone_bytes:
        return None

    pool: list[tuple[object, ...]] = []
    for _ in range(pool_size):
        pool.append(
            tuple(arg.clone() if is_tensor else arg for arg, is_tensor in zip(args, tensor_mask, strict=True))
        )
    return pool


def cuda_time_ms(
    fn: Callable[..., object],
    *,
    args: tuple[object, ...] = (),
    warmup: int = 10,
    iterations: int = 50,
    trials: int = 3,
    flush_l2: bool = True,
    clone_pool: bool = True,
    clone_pool_size: int = 3,
    clone_pool_max_bytes: int = 1 << 30,
) -> float:
    cache = _get_l2_flush_cache() if flush_l2 else None
    arg_pool = _build_arg_pool(args, pool_size=clone_pool_size, max_clone_bytes=clone_pool_max_bytes) if clone_pool else None

    def run_once() -> None:
        if cache is not None:
            cache.zero_()
        if arg_pool is None:
            fn(*args)
        else:
            current_args = arg_pool[run_once.iteration % len(arg_pool)]
            fn(*current_args)

    run_once.iteration = 0  # type: ignore[attr-defined]

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    trial_means: list[float] = []

    def on_trace_ready(prof) -> None:
        total_ms = _sum_cuda_kernel_ms(prof.profiler.kineto_results)
        trial_means.append(total_ms / iterations)

    try:
        schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=trials)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            acc_events=True,
        ) as prof:
            for _ in range(trials):
                for _ in range(iterations):
                    run_once.iteration += 1  # type: ignore[attr-defined]
                    run_once()
                prof.step()
                for _ in range(iterations):
                    run_once.iteration += 1  # type: ignore[attr-defined]
                    run_once()
                prof.step()
    except (AttributeError, RuntimeError):
        pass

    if trial_means and not any(total_ms > 0 for total_ms in trial_means):
        trial_means.clear()

    if not trial_means:
        for _ in range(trials):
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            for idx in range(iterations):
                if cache is not None:
                    cache.zero_()
                start_events[idx].record()
                if arg_pool is None:
                    fn(*args)
                else:
                    fn(*arg_pool[idx % len(arg_pool)])
                end_events[idx].record()
            torch.cuda.synchronize()
            times = [start.elapsed_time(end) for start, end in zip(start_events, end_events, strict=True)]
            trial_means.append(sum(times) / len(times))

    trial_means.sort()
    return trial_means[len(trial_means) // 2]
