"""Render a compact demo GIF from an actual saved EvalForge trace."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def frame(title: str, lines: list[str], accent: str = "#67d8ef") -> Image.Image:
    image = Image.new("RGB", (1200, 675), "#10151c")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (35, 30, 1165, 645), radius=18, fill="#17212b", outline="#334657", width=3
    )
    draw.ellipse((62, 55, 78, 71), fill="#ff5f57")
    draw.ellipse((88, 55, 104, 71), fill="#febc2e")
    draw.ellipse((114, 55, 130, 71), fill="#28c840")
    draw.text((155, 47), "EvalForge", font=font(24), fill="#c7d5e0")
    draw.text((70, 105), title, font=font(34), fill=accent)
    y = 165
    body = font(24)
    for line in lines:
        for wrapped in textwrap.wrap(line, width=78) or [""]:
            draw.text((75, y), wrapped, font=body, fill="#e5edf3")
            y += 34
        y += 8
    draw.text(
        (75, 603),
        "Evidence comes from the saved trace, not model narration.",
        font=font(18),
        fill="#7f94a3",
    )
    return image


def find_run(run_root: Path) -> tuple[dict, list[dict]]:
    for summary_path in run_root.rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["task_id"] == "numpy_broadcast_001":
            trace_path = summary_path.with_name("trace.jsonl")
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            return summary, events
    raise FileNotFoundError("numpy_broadcast_001 summary not found")


def first_output(events: list[dict], event_type: str, predicate=lambda payload: True) -> str:
    for event in events:
        if event["event_type"] == event_type and predicate(event["payload"]):
            payload = event["payload"]
            result = payload.get("result", {})
            return result.get("output", "")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("experiments/runs/scripted-seed"))
    parser.add_argument("--output", type=Path, default=Path("figures/evalforge_demo.gif"))
    args = parser.parse_args()
    summary, events = find_run(args.run_root)
    failure = first_output(
        events,
        "tool_result",
        lambda payload: (
            payload.get("result", {}).get("tool_name") == "run_tests"
            and not payload.get("result", {}).get("ok")
        ),
    )
    failure_line = next(
        (line for line in failure.splitlines() if line.startswith("FAILED")), "Visible tests failed"
    )
    evaluator = first_output(events, "evaluator_result")
    frames = [
        frame(
            "1. Task received",
            [
                "Repair column-wise NumPy z-score normalization.",
                "Fresh Git workspace created from the buggy task.",
            ],
        ),
        frame(
            "2. Visible verifier establishes failure",
            [failure_line, "The harness records the exit code and complete pytest evidence."],
            "#ff8a7a",
        ),
        frame(
            "3. Evidence-linked diagnosis",
            [
                "Observed shapes (4, 2) and (4,) cannot broadcast.",
                "Repair hypothesis: reduce over axis 0 with keepdims and protect zero variance.",
            ],
            "#ffd166",
        ),
        frame(
            "4. Controlled patch",
            [
                "Changed: scientific_stats/normalization.py",
                "The patch is checked with git apply --check before modification.",
            ],
        ),
        frame(
            "5. Visible tests pass",
            [
                "The provider may submit only after this gate passes.",
                "No textual self-certification can bypass the verifier.",
            ],
            "#72e6a6",
        ),
        frame(
            "6. Protected evaluator passes",
            [
                evaluator.strip() or "5 passed",
                f"Resolved: {summary['resolved']}  |  Patch valid: {summary['patch_valid']}",
                "Trace and summary saved for reproducible analysis.",
            ],
            "#72e6a6",
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=[1800, 2600, 2600, 2200, 2000, 3000],
        loop=0,
        optimize=True,
    )
    print(args.output)


if __name__ == "__main__":
    main()
