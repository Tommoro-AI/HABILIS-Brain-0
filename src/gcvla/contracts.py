from __future__ import annotations


def assert_no_future_gc_at_inference(batch: dict) -> None:
    forbidden = {"future_depth", "future_rgb", "gc_target_codes", "geometry_target"}
    leaked = sorted(forbidden.intersection(batch))
    if leaked:
        raise RuntimeError(f"future-derived GC inputs are forbidden at inference: {leaked}")
