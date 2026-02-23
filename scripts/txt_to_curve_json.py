#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_line(line: str, line_no: int) -> tuple[float, float] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    data_part = stripped.split("#", 1)[0].strip()
    if not data_part:
        return None

    tokens = [tok for tok in re.split(r"[\s,]+", data_part) if tok]
    if len(tokens) < 2:
        raise ValueError(f"line {line_no}: expected at least 2 columns, got '{line.rstrip()}'")

    try:
        x_val = float(tokens[0])
        y_val = float(tokens[1])
    except ValueError as exc:
        raise ValueError(f"line {line_no}: cannot parse float values: '{line.rstrip()}'") from exc

    return x_val, y_val


def convert_txt_to_curve_json(input_path: Path, output_path: Path) -> dict[str, list[float]]:
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    pairs: list[tuple[float, float]] = []
    with input_path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            parsed = _parse_line(line, line_no)
            if parsed is not None:
                pairs.append(parsed)

    if not pairs:
        raise ValueError(f"no valid data rows found in: {input_path}")

    pairs.sort(key=lambda item: item[0])
    payload = {
        "x": [x for x, _ in pairs],
        "y": [y for _, y in pairs],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert 2-column TXT data into Jarvis curve JSON format.",
    )
    parser.add_argument("input", help="Input TXT path (2 numeric columns: x y).")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. Default: <input_stem>.json in same directory.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.with_suffix(".json")

    payload = convert_txt_to_curve_json(input_path, output_path)
    print(f"Converted {len(payload['x'])} rows")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
