from __future__ import annotations

import json
from pathlib import Path
import re

from jarvis_operas import UNIT_CONTRACT_VERSION
from scripts.check_release import VERSION_PATTERN, _project_fields


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_public_unit_contract_and_builtin_units_are_documented() -> None:
    contract = (REPO_ROOT / "user-guide" / "unit_contract.md").read_text(encoding="utf-8")
    builtin_units = (REPO_ROOT / "user-guide" / "builtin_units.md").read_text(encoding="utf-8")

    assert "OperaFunction.metadata.unit_contract" in contract
    assert "JSON Pointer" in contract
    assert "strict=True" in contract
    assert "helper.eggbox" in builtin_units
    assert "pdg.mZ" in builtin_units
    assert "dmddxn" in builtin_units
    assert "dmddxe" in builtin_units


def test_public_schema_describes_the_unit_contract_without_release_state() -> None:
    schema = json.loads((REPO_ROOT / "user-guide" / "unit_contract_schema.json").read_text(encoding="utf-8"))

    assert schema["properties"]["version"]["const"] == UNIT_CONTRACT_VERSION
    assert {"inputs", "outputs"} <= schema["properties"].keys()
    assert "unit_spec" in schema["$defs"]
    assert "package_version" not in json.dumps(schema)


def test_public_migration_and_changelog_are_available() -> None:
    migration = (REPO_ROOT / "user-guide" / "migration.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    version = _project_fields((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["version"]

    assert "Import Path Updates" in migration
    assert "Downstream (Jarvis-HEP / Jarvis-PLOT) Adapter Checklist" in migration
    releases = re.findall(rf"(?m)^##[ \t]+v?({VERSION_PATTERN})(?=[ \t—–:-]|$)", changelog)
    assert releases and releases[0] == version
