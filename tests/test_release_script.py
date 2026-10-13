from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_release_entrypoint_uses_the_canonical_name() -> None:
    assert (ROOT / "jorel").is_file()
    assert os.access(ROOT / "jorel", os.X_OK)
    assert not (ROOT / "joerl").exists()


@pytest.fixture
def wrapper_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Use the real wrapper with an inert delegate, without releasing anything."""
    repository = tmp_path / "release repository"
    (repository / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "jorel", repository / "jorel")
    delegate = repository / "scripts" / "full-release.sh"
    delegate.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$PWD" > "$RELEASE_WRAPPER_LOG"\n'
        'printf "%s\\n" "$@" >> "$RELEASE_WRAPPER_LOG"\n',
        encoding="utf-8",
    )
    delegate.chmod(0o755)
    caller = tmp_path / "caller directory"
    caller.mkdir()
    return repository, caller


@pytest.mark.parametrize("shell", ["bash", "zsh"])
def test_sourced_wrapper_resolves_repository_and_forwards_arguments(
    wrapper_repository: tuple[Path, Path], shell: str
) -> None:
    shell_path = shutil.which(shell)
    if shell_path is None:
        pytest.skip(f"{shell} is unavailable")
    repository, caller = wrapper_repository
    log = caller / "wrapper.log"
    env = dict(os.environ, RELEASE_WRAPPER_LOG=str(log))
    # Passing paths as positional parameters avoids shell interpolation.
    completed = subprocess.run(
        [shell_path, "-c", 'source "$1"; shift; jorel "$@"', "release-test",
         str(repository / "jorel"), "1.5.1", "--output", "output directory", "--dry"],
        cwd=caller,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert log.read_text().splitlines() == [
        str(caller), "1.5.1", "--output", "output directory", "--dry"
    ]


def test_wrapper_can_be_executed_directly(
    wrapper_repository: tuple[Path, Path],
) -> None:
    repository, caller = wrapper_repository
    log = caller / "wrapper.log"
    completed = subprocess.run(
        [str(repository / "jorel"), "--help"],
        cwd=caller,
        env=dict(os.environ, RELEASE_WRAPPER_LOG=str(log)),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert log.read_text().splitlines() == [str(caller), "--help"]


FAKE_PYTHON = r'''
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile
import re

REAL_PYTHON = __REAL_PYTHON__
arguments = sys.argv[1:]
executable = Path(sys.argv[0]).absolute()
real_python = executable.with_name('python-real')
runner = str(real_python) if real_python.exists() else REAL_PYTHON
record = {'args': arguments, 'cwd': os.getcwd(), 'executable': str(executable)}
failure = os.environ.get('FAKE_RELEASE_FAIL', '')
normalized = [argument for argument in arguments if argument != '-I']
stdin_source = sys.stdin.read() if normalized and normalized[0] == '-' else None
stage = 'python'
if normalized[:2] == ['-m', 'venv']:
    stage = 'venv'
elif normalized[:2] == ['-m', 'build']:
    stage = 'build'
elif normalized[:3] == ['-m', 'twine', 'check']:
    stage = 'twine-check'
elif normalized[:3] == ['-m', 'twine', 'upload']:
    stage = 'upload'
elif normalized[:3] == ['-m', 'pip', 'check']:
    stage = 'pip-check'
elif normalized[:3] == ['-m', 'pip', 'install']:
    stage = 'pip-install'
elif normalized[:2] == ['-m', 'pytest']:
    stage = 'source-tests'
elif normalized and Path(normalized[0]).name == 'smoke_install.py':
    stage = 'installed-smoke'
elif normalized and Path(normalized[0]).name == 'generate_function_docs.py':
    stage = 'docs'
elif normalized and Path(normalized[0]).name == 'check_release.py':
    stage = 'artifact-check' if '--artifacts' in normalized else 'source-check'
elif stdin_source and 'release-verification.json' in stdin_source:
    stage = 'report'
elif normalized[:1] == ['-c'] and 'import build, twine, pytest, polars' in normalized[1]:
    stage = 'tooling'

record['stage'] = stage
if stage == 'build':
    source = Path(arguments[-1])
    record['snapshot_files'] = sorted(p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file())
    record['snapshot_readme'] = (source / 'README.md').read_text()
    record['find_links'] = os.environ.get('PIP_FIND_LINKS')
    fingerprint = hashlib.sha256()
    for path in sorted(source.rglob('*')):
        if path.is_file():
            fingerprint.update(path.relative_to(source).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    record['snapshot_sha256'] = fingerprint.hexdigest()
with open(os.environ['FAKE_RELEASE_LOG'], 'a') as log:
    log.write(json.dumps(record) + '\n')
if stage == failure:
    print('Simulated failure: ' + stage, file=sys.stderr)
    raise SystemExit(17)

if stage == 'venv':
    target = Path(arguments[-1])
    subprocess.run([REAL_PYTHON, '-m', 'venv', '--without-pip', '--system-site-packages', str(target)], check=True)
    (target / 'bin' / 'python').rename(target / 'bin' / 'python-real')
    shutil.copy2(executable, target / 'bin' / 'python')
elif stage == 'build':
    source = Path(arguments[-1])
    out = Path(arguments[arguments.index('--outdir') + 1])
    out.mkdir(parents=True)
    text = (source / 'pyproject.toml').read_text()
    # The fixture uses static strings/lists, so the test launcher also works on 3.10.
    table = re.search(r'(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)', text)[1]
    project = {
        field: ast.literal_eval(re.search(r'(?m)^' + field + r'\s*=\s*("[^"\n]*")', table)[1])
        for field in ('name', 'version', 'requires-python')
    }
    project['dependencies'] = ast.literal_eval(re.search(r'(?ms)^dependencies\s*=\s*(\[.*?\])', table)[1])
    optional = re.search(r'(?ms)^\[project.optional-dependencies\]\s*(.*?)(?=^\[|\Z)', text)
    project['optional-dependencies'] = {
        extra: ast.literal_eval(values)
        for extra, values in re.findall(r'(?ms)^([\w-]+)\s*=\s*(\[.*?\])', optional[1] if optional else '')
    }
    name = project['name'].replace('-', '_').lower()
    stem = name + '-' + project['version']
    metadata = 'Metadata-Version: 2.1\nName: ' + project['name'] + '\nVersion: ' + project['version'] + '\n'
    metadata += 'Requires-Python: ' + project['requires-python'] + '\n'
    for dependency in project.get('dependencies', []):
        metadata += 'Requires-Dist: ' + dependency + '\n'
    for extra, dependencies in project.get('optional-dependencies', {}).items():
        metadata += 'Provides-Extra: ' + extra + '\n'
        for dependency in dependencies:
            metadata += 'Requires-Dist: ' + dependency + '; extra == "' + extra + '"\n'
    with zipfile.ZipFile(out / (stem + '-py3-none-any.whl'), 'w') as wheel:
        for path in (source / 'jarvis_operas').rglob('*'):
            if path.is_file():
                wheel.write(path, path.relative_to(source).as_posix())
        wheel.writestr(stem + '.dist-info/METADATA', metadata)
        wheel.writestr(stem + '.dist-info/entry_points.txt', '[console_scripts]\njopera = jarvis_operas.cli:main\n')
        wheel.writestr(stem + '.dist-info/WHEEL', 'Wheel-Version: 1.0\nGenerator: test harness\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        wheel.writestr(stem + '.dist-info/RECORD', '')
    (source / 'PKG-INFO').write_text(metadata)
    egg_info = source / 'Jarvis_Operas.egg-info'
    egg_info.mkdir()
    (egg_info / 'PKG-INFO').write_text(metadata)
    (egg_info / 'entry_points.txt').write_text('[console_scripts]\njopera = jarvis_operas.cli:main\n')
    with tarfile.open(out / (stem + '.tar.gz'), 'w:gz') as sdist:
        for path in source.rglob('*'):
            if path.is_file():
                sdist.add(path, arcname=stem + '/' + path.relative_to(source).as_posix())
elif stage == 'pip-install':
    # Simulate installing the built package only inside this disposable venv.
    for argument in arguments:
        if argument.endswith('.whl'):
            site = subprocess.check_output([runner, '-c', 'import sysconfig;print(sysconfig.get_path("purelib"))'], text=True).strip()
            with zipfile.ZipFile(argument) as wheel:
                wheel.extractall(site)
elif stage in {'twine-check', 'upload', 'pip-check', 'source-tests', 'installed-smoke', 'docs', 'tooling'}:
    pass
else:
    raise SystemExit(subprocess.run([runner, *arguments], input=stdin_source, text=True).returncode)
'''


@pytest.fixture
def release_repository(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Run the real shell, with no index access or mutation of a user environment."""
    repository = tmp_path / "repository"
    repository.mkdir()
    if (ROOT / ".git").exists():
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    else:
        # The release runner executes this suite inside its Git-free source snapshot.
        tracked = [
            path.relative_to(ROOT).as_posix()
            for path in ROOT.rglob("*")
            if path.is_file() and not any(
                part in {".git", "__pycache__", ".pytest_cache", "build", "dist"}
                or part.endswith(".egg-info")
                for part in path.relative_to(ROOT).parts
            )
        ]
    required = ["jorel", "scripts/full-release.sh", "scripts/check_release.py", "scripts/smoke_install.py"]
    for relative in sorted(set(filter(None, tracked)) | set(required)):
        source = ROOT / relative
        if not source.is_file():
            continue
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
         "-c", "user.name=Release Test", "-c", "user.email=release-test@example.invalid",
         "commit", "-q", "-m", "Disposable release fixture"],
        cwd=repository,
        check=True,
    )
    # The build must preserve current tracked edits and omit unrelated untracked files.
    with (repository / "README.md").open("a") as readme:
        readme.write("\nTracked release test edit.\n")
    (repository / "untracked-secret.txt").write_text("Never package this.\n")
    caller = tmp_path / "outside repository"
    caller.mkdir()
    launcher = tmp_path / "fake-python"
    launcher.write_text(
        f"#!{sys.executable}\n" + textwrap.dedent(FAKE_PYTHON).replace("__REAL_PYTHON__", repr(sys.executable)),
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    log = tmp_path / "release-flow.jsonl"
    return repository, caller, launcher, log


def _run_release(
    fixture: tuple[Path, Path, Path, Path], *arguments: str, failure: str = ""
) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    repository, caller, python, log = fixture
    completed = subprocess.run(
        ["bash", str(repository / "scripts" / "full-release.sh"),
         "--python", str(python), *arguments],
        cwd=caller,
        env=dict(os.environ, FAKE_RELEASE_LOG=str(log), FAKE_RELEASE_FAIL=failure),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return completed, records


@pytest.mark.parametrize("arguments", [
    ("--help",), ("1.5",), ("--unknown",), ("--upload",), ("--dry", "--upload"),
    ("--output",), ("--wheelhouse",), ("--no-index",),
])
def test_release_preflight_finishes_before_tooling_or_output_mutation(
    release_repository: tuple[Path, Path, Path, Path], arguments: tuple[str, ...]
) -> None:
    repository, caller, _, _ = release_repository
    original_project = (repository / "pyproject.toml").read_bytes()
    completed, records = _run_release(release_repository, *arguments)
    assert completed.returncode == (0 if arguments == ("--help",) else 2), completed.stdout + completed.stderr
    assert records == []
    if arguments == ("--help",):
        assert "--upload" not in completed.stdout
    assert (repository / "pyproject.toml").read_bytes() == original_project
    assert list(caller.iterdir()) == []


def test_dry_release_verifies_tracked_snapshot_without_upload(
    release_repository: tuple[Path, Path, Path, Path],
) -> None:
    _, caller, _, _ = release_repository
    completed, records = _run_release(release_repository, "--dry", "--output", "verified artifacts")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    stages = [record["stage"] for record in records]
    assert "upload" not in stages
    assert stages.index("build") < stages.index("artifact-check") < stages.index("source-tests")
    assert stages.index("source-tests") < stages.index("pip-check") < stages.index("installed-smoke")
    build = next(record for record in records if record["stage"] == "build")
    assert "Tracked release test edit." in build["snapshot_readme"]
    assert "untracked-secret.txt" not in build["snapshot_files"]
    output = caller / "verified artifacts"
    assert len(list(output.glob("*.whl"))) == len(list(output.glob("*.tar.gz"))) == 1
    report = json.loads((output / "release-verification.json").read_text())
    assert report["source_snapshot_sha256"] == build["snapshot_sha256"]
    assert (output / "SHA256SUMS").is_file()
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    installs = [record["args"] for record in records if record["stage"] == "pip-install" and any(arg.endswith(".whl") for arg in record["args"])]
    assert len(installs) == 2
    assert all("--no-deps" not in args and "-e" not in args for args in installs)


@pytest.mark.parametrize("failure", [
    "build", "twine-check", "artifact-check", "source-tests", "pip-check", "installed-smoke", "report"
])
def test_failed_verification_never_uploads_or_publishes_output(
    release_repository: tuple[Path, Path, Path, Path], failure: str
) -> None:
    _, caller, _, _ = release_repository
    completed, records = _run_release(
        release_repository, "--output", "verified artifacts", failure=failure
    )
    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert failure in [record["stage"] for record in records]
    assert not any(record["stage"] == "upload" for record in records)
    assert not (caller / "verified artifacts").exists()


def test_default_release_uploads_only_the_verified_wheel_and_sdist(
    release_repository: tuple[Path, Path, Path, Path],
) -> None:
    _, caller, _, _ = release_repository
    completed, records = _run_release(release_repository, "--output", "verified artifacts")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    upload = next(record for record in records if record["stage"] == "upload")
    assert upload is records[-1]
    assert "--non-interactive" in upload["args"]
    paths = [Path(argument) for argument in upload["args"] if argument.endswith((".whl", ".tar.gz"))]
    assert set(paths) == set((caller / "verified artifacts").glob("*.whl")) | set((caller / "verified artifacts").glob("*.tar.gz"))
    assert len(paths) == 2


def test_existing_output_is_preserved_and_rejected_before_version_change(
    release_repository: tuple[Path, Path, Path, Path],
) -> None:
    repository, caller, _, _ = release_repository
    output = caller / "existing output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("Preserve existing output.\n")
    project = (repository / "pyproject.toml").read_bytes()
    completed, records = _run_release(release_repository, "1.5.1", "--output", str(output))
    assert completed.returncode != 0
    assert marker.read_text() == "Preserve existing output.\n"
    assert (repository / "pyproject.toml").read_bytes() == project
    assert not any(record["stage"] in {"venv", "build", "upload"} for record in records)


def test_offline_options_reach_build_and_complete_dependency_install(
    release_repository: tuple[Path, Path, Path, Path],
) -> None:
    _, caller, _, _ = release_repository
    wheelhouse = caller / "dependency wheelhouse"
    wheelhouse.mkdir()
    completed, records = _run_release(
        release_repository, "--dry", "--no-isolation", "--no-index",
        "--wheelhouse", "dependency wheelhouse", "--output", "verified artifacts"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    build = next(record for record in records if record["stage"] == "build")
    assert "--no-isolation" in build["args"]
    installs = [record["args"] for record in records if record["stage"] == "pip-install"]
    assert installs
    for arguments in installs:
        assert "--no-index" in arguments
        assert arguments[arguments.index("--find-links") + 1] == str(wheelhouse)
        assert "--no-deps" not in arguments
    assert not any(record["stage"] == "upload" for record in records)


def test_requested_version_requires_changelog_before_changing_source(
    release_repository: tuple[Path, Path, Path, Path],
) -> None:
    repository, caller, _, _ = release_repository
    original = (repository / "pyproject.toml").read_bytes()
    completed, records = _run_release(release_repository, "1.5.1", "--output", "verified artifacts")
    assert completed.returncode != 0
    assert (repository / "pyproject.toml").read_bytes() == original
    assert not any(record["stage"] in {"venv", "build", "upload"} for record in records)
    assert not (caller / "verified artifacts").exists()


def test_requested_version_updates_source_and_handoff_after_changelog_preparation(
    release_repository: tuple[Path, Path, Path, Path],
) -> None:
    repository, caller, _, _ = release_repository
    changelog = repository / "docs" / "CHANGELOG.md"
    changelog.write_text(
        changelog.read_text().replace("# Changelog\n", "# Changelog\n\n## 1.5.1 — 2026-10-13\n\nPrepared test release.\n", 1)
    )
    completed, records = _run_release(release_repository, "1.5.1", "--dry", "--output", "verified artifacts")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'version = "1.5.1"' in (repository / "pyproject.toml").read_text()
    handoff = json.loads((repository / "docs" / "unit_contract_handoff.json").read_text())
    assert handoff["implementation"]["package_version"] == "1.5.1"
    assert len(list((caller / "verified artifacts").glob("*1.5.1*.whl"))) == 1
    assert not any(record["stage"] == "upload" for record in records)


def test_python_path_is_resolved_relative_to_the_caller(
    release_repository: tuple[Path, Path, Path, Path],
) -> None:
    _, caller, python, _ = release_repository
    completed, records = _run_release(
        release_repository, "--dry", "--python", os.path.relpath(python, caller),
        "--output", "verified artifacts"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert any(record["stage"] == "build" for record in records)
    assert (caller / "verified artifacts" / "release-verification.json").is_file()


def test_spaced_wheelhouse_reaches_isolated_build_as_a_uri(
    release_repository: tuple[Path, Path, Path, Path],
) -> None:
    _, caller, _, _ = release_repository
    wheelhouse = caller / "dependency wheelhouse"
    wheelhouse.mkdir()
    completed, records = _run_release(
        release_repository, "--dry", "--wheelhouse", "dependency wheelhouse", "--no-index",
        "--output", "verified artifacts"
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    build = next(record for record in records if record["stage"] == "build")
    assert "--no-isolation" not in build["args"]
    assert build["find_links"] == wheelhouse.resolve().as_uri()
    assert "%20" in build["find_links"]


@pytest.mark.parametrize("relative", [
    "jarvis_operas/models/cosmopower/CP_paper/CMB/README.md",
    "jarvis_operas/models/cosmopower/CP_paper/CMB/cmb_TT_NN.pkl",
    "jarvis_operas/namespaces/hts/defs/core.py",
])
def test_deleted_tracked_runtime_file_fails_before_build_or_upload(
    release_repository: tuple[Path, Path, Path, Path], relative: str
) -> None:
    repository, caller, _, _ = release_repository
    (repository / relative).unlink()
    completed, records = _run_release(release_repository, "--output", "verified artifacts")
    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert not any(record["stage"] in {"build", "upload"} for record in records)
    assert not (caller / "verified artifacts").exists()
