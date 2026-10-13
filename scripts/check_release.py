"""Read-only source, wheel and sdist release gates using only the standard library."""

from __future__ import annotations

import argparse
import configparser
from email.parser import BytesParser
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile


DISTRIBUTION = "Jarvis-Operas"
PACKAGE = "jarvis_operas"
ENTRY_POINT = "jarvis_operas.cli:main"
VERSION_PATTERN = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"


def _project_fields(text: str) -> dict[str, str]:
    """Read static project fields on Python 3.10 without a TOML dependency."""
    try:
        import tomllib
    except ImportError:
        tables = re.findall(r"(?ms)^\[project\][ \t]*(?:\#.*)?\n(.*?)(?=^\[|\Z)", text)
        if len(tables) != 1:
            raise ValueError("pyproject.toml must contain exactly one [project] table")
        dynamic = re.search(r"(?ms)^[ \t]*dynamic[ \t]*=[ \t]*\[(.*?)\]", tables[0])
        if dynamic and re.search(r"[\"']version[\"']", dynamic[1]):
            raise ValueError("pyproject.toml requires a static project.version")
        result = {}
        for name in ("name", "version"):
            matches = re.findall(
                rf"(?m)^[ \t]*{name}[ \t]*=[ \t]*([\"'])([^\"'\n]+)\1[ \t]*(?:\#.*)?$",
                tables[0],
            )
            assignments = re.findall(rf"(?m)^[ \t]*{name}[ \t]*=", tables[0])
            if len(matches) != 1 or len(assignments) != 1:
                raise ValueError(f"pyproject.toml requires one static project.{name}")
            result[name] = matches[0][1]
        return result
    else:
        try:
            project = tomllib.loads(text).get("project", {})
        except ValueError as exc:
            raise ValueError(f"Invalid pyproject.toml: {exc}") from exc
        if any(not isinstance(project.get(name), str) for name in ("name", "version")):
            raise ValueError("pyproject.toml requires static project.name and project.version")
        if "version" in project.get("dynamic", []):
            raise ValueError("pyproject.toml requires a static project.version")
        return {name: project[name] for name in ("name", "version")}


def check_source(root: Path, expected_version: str | None = None) -> str:
    """Require the current source version, changelog and unit handoff to agree."""
    root = Path(root)
    project = _project_fields((root / "pyproject.toml").read_text(encoding="utf-8"))
    if project["name"] != DISTRIBUTION:
        raise ValueError(f"Distribution name must be {DISTRIBUTION}")
    version = project["version"]
    if re.fullmatch(VERSION_PATTERN, version) is None:
        raise ValueError("Release version must be a static X.Y.Z version")
    if expected_version is not None and expected_version != version:
        raise ValueError(f"Expected version {expected_version!r} disagrees with source {version}")
    changelog = (root / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"(?m)^##[ \t]+(.+)$", changelog)
    version_heading = re.compile(rf"^v?({VERSION_PATTERN})(?=[ \t—–:-]|$)")
    releases = [match[1] for heading in headings if (match := version_heading.match(heading))]
    if not releases or releases[0] != version:
        raise ValueError(f"CHANGELOG.md: latest release entry must be {version}")
    handoff = json.loads((root / "docs" / "unit_contract_handoff.json").read_text(encoding="utf-8"))
    implementation = handoff.get("implementation", {}) if isinstance(handoff, dict) else {}
    if not isinstance(implementation, dict):
        raise ValueError("Unit handoff implementation must be an object")
    if implementation.get("package") != DISTRIBUTION or implementation.get("package_version") != version:
        raise ValueError("Unit handoff package name/version disagrees with source")
    return version


def _runtime_file(name: str) -> bool:
    path = PurePosixPath(name)
    if not path.parts or path.parts[0] != PACKAGE:
        return False
    return (
        path.suffix == ".py"
        or (len(path.parts) > 1 and path.parts[1] == "manifests" and path.suffix == ".json")
        or (len(path.parts) > 1 and path.parts[1] == "models" and path.suffix in {".pkl", ".md", ".txt"})
    )


def expected_runtime_files(root: Path) -> dict[str, bytes]:
    """Use Git's tracked inventory in checkouts and native files in source archives."""
    root = Path(root).resolve()
    if (root / ".git").exists():
        process = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", PACKAGE],
            check=True, capture_output=True, timeout=30,
        )
        names = [name.decode("utf-8") for name in process.stdout.split(b"\0") if name]
    else:
        names = [path.relative_to(root).as_posix() for path in (root / PACKAGE).rglob("*") if path.is_file()]
    expected = {name: (root / name).read_bytes() for name in names if _runtime_file(name)}
    if f"{PACKAGE}/__init__.py" not in expected:
        raise ValueError("Source package is missing jarvis_operas/__init__.py")
    return expected


def _check_metadata(payload: bytes, version: str, label: str) -> None:
    metadata = BytesParser().parsebytes(payload)
    if metadata.get_all("Name") != [DISTRIBUTION] or metadata.get_all("Version") != [version]:
        raise ValueError(f"{label} name/version disagrees with source")


def _check_entrypoint(payload: bytes, label: str) -> None:
    config = configparser.ConfigParser(interpolation=None)
    try:
        config.read_string(payload.decode("utf-8"))
    except (configparser.Error, UnicodeError) as exc:
        raise ValueError(f"{label} has invalid entry point metadata") from exc
    if config.get("console_scripts", "jopera", fallback=None) != ENTRY_POINT:
        raise ValueError(f"{label} is missing the official jopera entry point")


def _check_runtime(expected: dict[str, bytes], files: dict[str, bytes], label: str) -> None:
    for name, payload in expected.items():
        if name not in files:
            raise ValueError(f"{label} is missing {name}")
        if files[name] != payload:
            raise ValueError(f"{label} has stale content: {name}")
    extras = sorted(name for name in files if _runtime_file(name) and name not in expected)
    if extras:
        raise ValueError(f"{label} contains unexpected package content: {extras[0]}")


def _archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def check_wheel(root: Path, wheel: Path, version: str) -> dict[str, object]:
    expected = expected_runtime_files(root)
    with zipfile.ZipFile(wheel) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        if len(names) != len(set(names)) or any(not _archive_name(name) for name in names):
            raise ValueError("Wheel contains duplicate or unsafe archive paths")
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Wheel must contain one distribution metadata file")
        prefix = metadata_paths[0].rsplit("/", 1)[0]
        if any(PurePosixPath(name).parts[0] not in {PACKAGE, prefix} for name in names):
            raise ValueError("Wheel contains unrelated packages or top-level files")
        files = {name: archive.read(name) for name in names}
        _check_metadata(files[metadata_paths[0]], version, "Wheel")
        entrypoint = prefix + "/entry_points.txt"
        if entrypoint not in files:
            raise ValueError("Wheel is missing entry point metadata")
        _check_entrypoint(files[entrypoint], "Wheel")
        _check_runtime(expected, files, "Wheel")
    return {"wheel": str(wheel), "wheel_runtime_files": len(expected)}


def check_sdist(root: Path, sdist: Path, version: str) -> dict[str, object]:
    expected = expected_runtime_files(root)
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        if any(not _archive_name(item.name) or not (item.isfile() or item.isdir()) for item in members):
            raise ValueError("Sdist contains unsafe paths or non-regular files")
        roots = {PurePosixPath(item.name).parts[0] for item in members}
        if len(roots) != 1:
            raise ValueError("Sdist must contain one top-level directory")
        files = {}
        for item in members:
            if item.isfile():
                name = PurePosixPath(item.name).relative_to(next(iter(roots))).as_posix()
                if name in files:
                    raise ValueError("Sdist contains duplicate archive paths")
                stream = archive.extractfile(item)
                assert stream is not None
                files[name] = stream.read()
        if "PKG-INFO" not in files:
            raise ValueError("Sdist is missing distribution metadata")
        _check_metadata(files["PKG-INFO"], version, "Sdist")
        info_paths = [name for name in files if name.endswith(".egg-info/PKG-INFO")]
        if len(info_paths) != 1:
            raise ValueError("Sdist must contain one egg-info metadata file")
        prefix = info_paths[0].rsplit("/", 1)[0]
        _check_metadata(files[info_paths[0]], version, "Sdist egg-info")
        entrypoint = prefix + "/entry_points.txt"
        if entrypoint not in files:
            raise ValueError("Sdist is missing entry point metadata")
        _check_entrypoint(files[entrypoint], "Sdist")
        for name in ("pyproject.toml",):
            if name not in files or files[name] != (Path(root) / name).read_bytes():
                raise ValueError(f"Sdist has missing or stale content: {name}")
        _check_runtime(expected, files, "Sdist")
        # Tests/scripts may be Python source, but another importable package is not ours.
        other_packages = [name for name in files if name.endswith("/__init__.py")
                          and PurePosixPath(name).parts[0] not in {PACKAGE, "tests", "scripts"}]
        if other_packages:
            raise ValueError(f"Sdist contains unrelated package: {other_packages[0]}")
    return {"sdist": str(sdist), "sdist_runtime_files": len(expected)}


def check_artifacts(root: Path, directory: Path, version: str) -> dict[str, object]:
    directory = Path(directory)
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Artifact directory must contain exactly one wheel and one sdist")
    return {**check_wheel(root, wheels[0], version), **check_sdist(root, sdists[0], version), "artifacts": "passed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--expected-version", help="Require this exact source and artifact version")
    parser.add_argument("--artifacts", type=Path, help="Directory containing exactly one wheel and one sdist")
    args = parser.parse_args(argv)
    try:
        version = check_source(args.root, args.expected_version)
        report = {"version": version, "source": "passed", "publishes": False}
        if args.artifacts is not None:
            report.update(check_artifacts(args.root, args.artifacts, version))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, zipfile.BadZipFile, tarfile.TarError) as exc:
        parser.exit(1, f"Release check failed: {exc}\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
