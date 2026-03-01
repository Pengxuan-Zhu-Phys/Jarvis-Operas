from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_generate_function_docs_script(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "generate_function_docs.py"
    out_dir = tmp_path / "func_docs"

    subprocess.run(
        [sys.executable, str(script), "--out", str(out_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (out_dir / "index.md").is_file()
    assert (out_dir / "functions.json").is_file()

    index_payload = json.loads((out_dir / "functions.json").read_text(encoding="utf-8"))
    assert isinstance(index_payload, list)
    assert any(item["full_name"] == "math.add" for item in index_payload)
    assert any(item["full_name"] == "interp1.interp1_xy_flat" for item in index_payload)
    assert any(item["full_name"] == "dmdd.LZSI2024" for item in index_payload)

    math_add_page = out_dir / "math" / "add.md"
    assert math_add_page.is_file()
    body = math_add_page.read_text(encoding="utf-8")
    assert "Metadata (from `OperaFunction.metadata`)" in body
    assert "summary" in body

    assert (out_dir / "interp1" / "interp1_xy_flat.md").is_file()
    assert (out_dir / "dmdd" / "LZSI2024.md").is_file()
