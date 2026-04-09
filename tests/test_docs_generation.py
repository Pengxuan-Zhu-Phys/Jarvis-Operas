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
    assert any(item["full_name"] == "dmddxn.LZSI2024" for item in index_payload)
    assert any(item["full_name"] == "dmddxn.XENONnTSI2025" for item in index_payload)
    assert any(item["full_name"] == "dmddxn.LZSDp2024" for item in index_payload)
    assert any(item["full_name"] == "dmddxn.XENONnTSDp2025Combined" for item in index_payload)
    assert any(item["full_name"] == "dmddxn.LZSDn2024" for item in index_payload)
    assert any(item["full_name"] == "dmddxn.XENONnTSDn2025" for item in index_payload)
    assert any(item["full_name"] == "dmddxn.CDMSSDn2010" for item in index_payload)
    assert any(item["full_name"] == "dmddxe.CDEXLM2022" for item in index_payload)
    assert any(item["full_name"] == "dmddxe.SENSEILM2024Combined" for item in index_payload)
    assert any(item["full_name"] == "dmddxe.DAMICMLM2025Combined" for item in index_payload)
    assert any(item["full_name"] == "dmddxe.AllLimitsLM2024" for item in index_payload)
    assert any(item["full_name"] == "dmddxe.XENON1THM2019" for item in index_payload)

    math_add_page = out_dir / "math" / "add.md"
    assert math_add_page.is_file()
    body = math_add_page.read_text(encoding="utf-8")
    assert "Metadata (from `OperaFunction.metadata`)" in body
    assert "summary" in body

    assert (out_dir / "interp1" / "interp1_xy_flat.md").is_file()
    assert (out_dir / "dmddxn" / "LZSI2024.md").is_file()
    assert (out_dir / "dmddxn" / "XENONnTSI2025.md").is_file()
    assert (out_dir / "dmddxn" / "PandaXSI2024.md").is_file()
    assert (out_dir / "dmddxn" / "LZSDp2024.md").is_file()
    assert (out_dir / "dmddxn" / "XENONnTSDp2025Combined.md").is_file()
    assert (out_dir / "dmddxn" / "LZSDn2024.md").is_file()
    assert (out_dir / "dmddxn" / "XENONnTSDn2025.md").is_file()
    assert (out_dir / "dmddxn" / "CDMSSDn2010.md").is_file()
    assert (out_dir / "dmddxe" / "index.md").is_file()
    assert (out_dir / "dmddxe" / "CDEXLM2022.md").is_file()
    assert (out_dir / "dmddxe" / "SENSEILM2024Combined.md").is_file()
    assert (out_dir / "dmddxe" / "DAMICMLM2025Combined.md").is_file()
    assert (out_dir / "dmddxe" / "AllLimitsLM2024.md").is_file()
    assert (out_dir / "dmddxe" / "XENON1THM2019.md").is_file()
