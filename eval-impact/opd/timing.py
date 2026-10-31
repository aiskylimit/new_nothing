"""Explicit, timezone-aware wall-clock budgets and rolling ETA estimates."""
from datetime import datetime, timezone


def deadline_timestamp(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("stop-at requires an explicit timezone, e.g. 2026-11-01T11:50:00+07:00")
    return parsed.timestamp()


def estimate_timing(step, total_steps, durations, now, deadline=None):
    mean = sum(durations) / len(durations)
    remaining = max(total_steps - step, 0)
    eta = remaining * mean
    available = max(deadline - now, 0.) if deadline is not None else None
    return {"rolling_step_seconds": mean, "eta_window_steps": len(durations),
            "eta_remaining_hours": eta / 3600,
            "estimated_finish_utc": datetime.fromtimestamp(now + eta, timezone.utc).isoformat(),
            "required_seconds_per_step": available / remaining if available is not None and remaining else None,
            "estimated_within_deadline": eta <= available if available is not None else None,
            "eta_excludes_checkpoint_io": True}
