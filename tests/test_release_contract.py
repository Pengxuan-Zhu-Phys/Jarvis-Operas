"""Release gates reject stale metadata and incomplete published runtime resources."""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts.check_release import (
    _project_fields,
    check_artifacts,
    check_sdist,
    check_source,
    check_wheel,
    main,
)


VERSION = "1.5.0"
RESOURCES = {
    "jarvis_operas/__init__.py": b"# Package fixture\n",
    "jarvis_operas/core/registry.py": b"# Native module fixture\n",
    "jarvis_operas/manifests/dmddxn/curve.json": b'{"unit":"cm^2"}\n',
    "jarvis_operas/models/cmb/planck/data/TT_power_spec.txt": b"2 100\n",
    "jarvis_operas/models/cmb/planck/data/README.md": b"# Dataset provenance\n",
    "jarvis_operas/models/cosmopower/CP_paper/CMB/cmb_TT_NN.pkl": b"model-fixture",
}


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools"]\n[project]\n'
        f'name = "Jarvis-Operas"\nversion = "{VERSION}"\n'
        '[project.optional-dependencies]\ndev = ["pytest"]\n', encoding="utf-8",
    )
    (root / "docs").mkdir()
    (root / "docs/CHANGELOG.md").write_text(f"# Changelog\n\n## {VERSION} — 2026-10-13\n\nRelease.\n")
    (root / "docs/unit_contract_handoff.json").write_text(json.dumps({
        "implementation": {"package": "Jarvis-Operas", "package_version": VERSION},
    }))
    for name, payload in RESOURCES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


def artifacts(root, *, wheel_damage=None, sdist_damage=None):
    target = root.parent / "dist"
    target.mkdir(exist_ok=True)
    metadata = f"Name: Jarvis-Operas\nVersion: {VERSION}\nProvides-Extra: dev\n".encode()
    entrypoint = b"[console_scripts]\njopera = jarvis_operas.cli:main\n"

    def damage(files, alteration):
        if alteration:
            action, name = alteration
            if action == "missing":
                del files[name]
            elif action == "stale":
                files[name] = b"stale source content"
            elif action == "version":
                files[name] = metadata.replace(VERSION.encode(), b"1.4.0")
            elif action == "name":
                files[name] = metadata.replace(b"Jarvis-Operas", b"Jarvis-HEP")
            elif action == "entrypoint":
                files[name] = b"[console_scripts]\njopera = wrong_package:main\n"
            elif action == "extra":
                files[name] = b"# unrelated source\n"
            else:
                raise AssertionError(action)

    wheel_files = {**RESOURCES,
                   f"jarvis_operas-{VERSION}.dist-info/METADATA": metadata,
                   f"jarvis_operas-{VERSION}.dist-info/entry_points.txt": entrypoint}
    damage(wheel_files, wheel_damage)
    wheel = target / f"jarvis_operas-{VERSION}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in wheel_files.items():
            archive.writestr(name, payload)
    sdist_files = {**RESOURCES, "pyproject.toml": (root / "pyproject.toml").read_bytes(),
                   "PKG-INFO": metadata, "Jarvis_Operas.egg-info/PKG-INFO": metadata,
                   "Jarvis_Operas.egg-info/entry_points.txt": entrypoint}
    damage(sdist_files, sdist_damage)
    sdist = target / f"jarvis_operas-{VERSION}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, payload in sdist_files.items():
            item = tarfile.TarInfo(f"jarvis_operas-{VERSION}/{name}")
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
    return target, wheel, sdist


def test_current_source_contract():
    root = Path(__file__).resolve().parents[1]
    project = _project_fields((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert check_source(root) == project["version"]


@pytest.mark.parametrize("version", ["1.5", "v1.5.0", "1.5.0rc1", "01.5.0"])
def test_source_rejects_non_release_versions(source, version):
    path = source / "pyproject.toml"
    path.write_text(path.read_text().replace(VERSION, version))
    with pytest.raises(ValueError, match="X.Y.Z"):
        check_source(source)


@pytest.mark.parametrize("declaration", ['dynamic = ["version"]', 'version = "1.5.0"\ndynamic = ["version"]'])
def test_source_rejects_missing_static_version(source, declaration):
    path = source / "pyproject.toml"
    path.write_text(path.read_text().replace(f'version = "{VERSION}"', declaration))
    with pytest.raises(ValueError, match="static"):
        check_source(source)


def test_source_rejects_distribution_and_requested_version_drift(source):
    with pytest.raises(ValueError, match="Expected version"):
        check_source(source, "1.4.0")
    path = source / "pyproject.toml"
    path.write_text(path.read_text().replace("Jarvis-Operas", "Jarvis-HEP"))
    with pytest.raises(ValueError, match="Distribution name"):
        check_source(source)


@pytest.mark.parametrize("heading", ["## 1.4.0 — today", "## Changes\n\nHistorical mention: 1.5.0", "## 1.5.1 — next\n\n## 1.5.0 — old"])
def test_source_requires_current_latest_changelog_heading(source, heading):
    (source / "docs/CHANGELOG.md").write_text(heading)
    with pytest.raises(ValueError, match="latest release entry"):
        check_source(source)


@pytest.mark.parametrize("damage", ["missing", "version", "package", "malformed"])
def test_source_requires_matching_machine_handoff(source, damage):
    path = source / "docs/unit_contract_handoff.json"
    if damage == "missing":
        path.unlink()
    elif damage == "malformed":
        path.write_text("broken json")
    else:
        handoff = json.loads(path.read_text())
        handoff["implementation"]["package_version" if damage == "version" else "package"] = "wrong"
        path.write_text(json.dumps(handoff))
    with pytest.raises((ValueError, OSError)):
        check_source(source)


def test_python310_project_fallback_is_scoped_and_static(monkeypatch):
    original = builtins.__import__

    def python310(name, *args, **kwargs):
        if name == "tomllib":
            raise ImportError("Python 3.10")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", python310)
    valid = '[build-system]\nversion="9.9.9"\n[project]\nname="Jarvis-Operas"\nversion="1.5.0" # release\n[tool.foo]\nversion="8.8.8"\n'
    assert _project_fields(valid) == {"name": "Jarvis-Operas", "version": VERSION}
    for invalid in (valid.replace('version="1.5.0" # release', 'dynamic=["version"]'),
                    valid + '[project]\nname="Jarvis-Operas"\nversion="1.4.0"\n',
                    valid.replace('version="1.5.0" # release', 'version="1.5.0"\nversion="1.4.0"')):
        with pytest.raises(ValueError):
            _project_fields(invalid)


@pytest.mark.parametrize("resource", list(RESOURCES))
@pytest.mark.parametrize("action", ["missing", "stale"])
def test_wheel_rejects_missing_and_stale_runtime_content(source, resource, action):
    _, wheel, _ = artifacts(source, wheel_damage=(action, resource))
    with pytest.raises(ValueError, match="missing|stale"):
        check_wheel(source, wheel, VERSION)


@pytest.mark.parametrize("resource", ["jarvis_operas/core/registry.py", "jarvis_operas/manifests/dmddxn/curve.json", "jarvis_operas/models/cosmopower/CP_paper/CMB/cmb_TT_NN.pkl"])
@pytest.mark.parametrize("action", ["missing", "stale"])
def test_sdist_rejects_missing_and_stale_runtime_content(source, resource, action):
    _, _, sdist = artifacts(source, sdist_damage=(action, resource))
    with pytest.raises(ValueError, match="missing|stale"):
        check_sdist(source, sdist, VERSION)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("action", ["version", "name", "entrypoint", "extra"])
def test_artifacts_reject_metadata_entrypoint_and_unrelated_package_drift(source, kind, action):
    if action == "extra":
        name = "unrelated/__init__.py"
    elif kind == "wheel":
        name = f"jarvis_operas-{VERSION}.dist-info/" + ("entry_points.txt" if action == "entrypoint" else "METADATA")
    else:
        name = "Jarvis_Operas.egg-info/entry_points.txt" if action == "entrypoint" else "PKG-INFO"
    directory, _, _ = artifacts(source, **{kind + "_damage": (action, name)})
    with pytest.raises(ValueError, match="name/version|entry point|unrelated"):
        check_artifacts(source, directory, VERSION)


def test_artifact_gate_requires_one_wheel_and_sdist(source):
    directory, wheel, sdist = artifacts(source)
    sdist.unlink()
    with pytest.raises(ValueError, match="exactly one"):
        check_artifacts(source, directory, VERSION)
    directory, wheel, _ = artifacts(source)
    (directory / "stale.whl").write_bytes(wheel.read_bytes())
    with pytest.raises(ValueError, match="exactly one"):
        check_artifacts(source, directory, VERSION)


def test_gates_are_read_only_and_allow_dev_extra(source, capsys):
    directory, wheel, sdist = artifacts(source)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    assert check_source(source, VERSION) == VERSION
    assert check_wheel(source, wheel, VERSION)["wheel_runtime_files"] == len(RESOURCES)
    assert check_sdist(source, sdist, VERSION)["sdist_runtime_files"] == len(RESOURCES)
    assert main(["--root", str(source), "--expected-version", VERSION, "--artifacts", str(directory)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["publishes"] is False and report["artifacts"] == "passed"
    assert before == {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    with pytest.raises(SystemExit) as failure:
        main(["--root", str(source), "--expected-version", "1.4.0"])
    assert failure.value.code == 1
