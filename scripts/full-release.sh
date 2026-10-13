#!/usr/bin/env bash
# Build, verify and publish one complete Operas wheel/sdist; --dry skips upload.
set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${JARVIS_RELEASE_PYTHON:-python3}"
requested_version=""
output=""
wheelhouse=""
dry=0
no_isolation=0
no_index=0

usage() {
  cat <<'EOF'
Usage: jorel [X.Y.Z] [--dry] [--output DIR] [--python PYTHON]
             [--no-isolation] [--wheelhouse DIR [--no-index]]

Default: build wheel/sdist, run tests, install all runtime dependencies in a
clean venv, verify the installed CLI/unit API, and write a verification report.
Then publish the exact verified wheel and sdist to PyPI.
--dry             Only build and verify; skip publication.
--output DIR      Fresh artifact directory (default: repository/dist/VERSION).
--python PYTHON   Python executable (default: JARVIS_RELEASE_PYTHON or python3).
--no-isolation    Reuse installed build/test tools in a separate tooling venv.
--wheelhouse DIR  Local wheels used by tooling, builds and complete installation.
--no-index        Resolve only from --wheelhouse; requires a complete wheelhouse.

Source this helper in Bash/Zsh to use jorel from any directory.
No caller-environment reinstall, package deletion, Git commit, tag, or push.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry) dry=1; shift ;;
    --no-isolation) no_isolation=1; shift ;;
    --no-index) no_index=1; shift ;;
    --output|--python|--wheelhouse)
      option="$1"
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $option" >&2; exit 2; }
      case "$option" in
        --output) output="$2" ;;
        --python) python_bin="$2" ;;
        --wheelhouse) wheelhouse="$2" ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      [[ -z "$requested_version" && "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
        echo "Unknown argument or invalid version: $1 (expected X.Y.Z)" >&2; exit 2;
      }
      requested_version="$1"; shift ;;
  esac
done
[[ "$no_index" -eq 0 || -n "$wheelhouse" ]] || { echo '--no-index requires --wheelhouse' >&2; exit 2; }
if [[ "$python_bin" == */* && "$python_bin" != /* ]]; then python_bin="$PWD/$python_bin"; fi
command -v "$python_bin" >/dev/null || { echo "Python executable not found: $python_bin" >&2; exit 2; }

# Resolve caller-relative paths before entering the repository.
if [[ -n "$output" && "$output" != /* ]]; then output="$PWD/$output"; fi
if [[ -n "$wheelhouse" && "$wheelhouse" != /* ]]; then wheelhouse="$PWD/$wheelhouse"; fi
if [[ -n "$wheelhouse" ]]; then
  [[ -d "$wheelhouse" ]] || { echo "Wheelhouse not found: $wheelhouse" >&2; exit 2; }
  wheelhouse="$(cd "$wheelhouse" && pwd)"
fi
cd "$release_root"

# Compute the requested version and validate public release notes before writes.
version="$("$python_bin" - "$requested_version" <<'PY'
from pathlib import Path
import re
import sys
from scripts.check_release import DISTRIBUTION, VERSION_PATTERN, _project_fields, check_public_documents
if sys.version_info < (3, 10):
    raise SystemExit('Release preparation requires Python 3.10 or newer')
text = Path('pyproject.toml').read_text(encoding='utf-8')
project = _project_fields(text)
if project['name'] != DISTRIBUTION:
    raise SystemExit('Distribution name must be Jarvis-Operas')
version = sys.argv[1] or project['version']
if not re.fullmatch(VERSION_PATTERN, version):
    raise SystemExit('Expected an X.Y.Z release version')
try:
    check_public_documents(Path('.'), version)
except (OSError, ValueError) as exc:
    raise SystemExit(f'Release check failed: {exc}') from exc
print(version)
PY
)"
output="${output:-$release_root/dist/$version}"
if [[ -e "$output" ]]; then
  [[ -d "$output" && -z "$(ls -A "$output")" ]] || {
    echo "Output must be a fresh empty directory: $output" >&2; exit 2;
  }
fi

"$python_bin" - "$version" <<'PY'
from pathlib import Path
import re
import sys
version = sys.argv[1]
path = Path('pyproject.toml')
text = path.read_text(encoding='utf-8')
project = re.search(r'(?ms)^[ \t]*\[project\][ \t]*(?:#.*)?\n(.*?)(?=^[ \t]*\[|\Z)', text)
pattern = r'''(?m)^([ \t]*version[ \t]*=[ \t]*)(["'])([^"'\n]+)\2([ \t]*(?:#.*)?)$'''
updated, count = re.subn(pattern, lambda m: m[1] + m[2] + version + m[2] + m[4], project[1], count=1)
if count != 1:
    raise SystemExit('Cannot update project.version')
new_text = text[:project.start(1)] + updated + text[project.end(1):]
if new_text != text:
    path.write_text(new_text, encoding='utf-8')
    print(f'Updated source version to {version}', file=sys.stderr)
PY
"$python_bin" scripts/check_release.py --root "$release_root" --expected-version "$version"

work="$(mktemp -d "${TMPDIR:-/tmp}/jarvis-operas-release.XXXXXXXX")"
trap 'rm -rf "$work"' EXIT
pip_options=(--disable-pip-version-check)
if [[ -n "$wheelhouse" ]]; then
  pip_options+=(--find-links "$wheelhouse")
  # pip parses this environment variable as whitespace-delimited URLs.
  PIP_FIND_LINKS="$("$python_bin" - "$wheelhouse" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve().as_uri())
PY
)"
  export PIP_FIND_LINKS
fi
if [[ "$no_index" -eq 1 ]]; then
  pip_options+=(--no-index)
  export PIP_NO_INDEX=1
fi
unset PYTHONPATH

printf '[jorel] Preparing Jarvis-Operas %s with %s\n' "$version" "$python_bin"
if [[ "$no_isolation" -eq 1 ]]; then
  "$python_bin" -m venv --system-site-packages "$work/tools"
else
  "$python_bin" -m venv "$work/tools"
fi
build_python="$work/tools/bin/python"
if [[ "$no_isolation" -eq 1 ]]; then
  "$build_python" -c 'import build, twine, pytest, polars; print("Existing build/test tools available")'
else
  "$build_python" -m pip install "${pip_options[@]}" build twine pytest polars
fi

# Current tracked edits are intentional inputs. Do not sweep unrelated untracked code.
"$build_python" - "$release_root" "$work/source" <<'PY'
from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
from scripts.check_release import expected_runtime_files, is_private_document
root, target = map(Path, sys.argv[1:])
# Missing tracked native modules/resources must not silently vanish from a snapshot.
expected_runtime_files(root)
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
required = ['jorel', 'scripts/full-release.sh', 'scripts/check_release.py', 'scripts/smoke_install.py', 'CHANGELOG.md']
source_hash = hashlib.sha256()
# Private docs are never release inputs, including stale tracked index entries.
for relative in sorted(set(filter(None, tracked)) | set(required)):
    if is_private_document(relative):
        continue
    source = root / relative
    if source.is_file():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash.update(relative.encode() + b'\0' + destination.read_bytes() + b'\0')
for relative in required:
    if not (target / relative).is_file():
        raise SystemExit(f'Missing release helper: {relative}')
(target.parent / 'source-snapshot.sha256').write_text(source_hash.hexdigest() + '\n', encoding='utf-8')
PY
source_root="$work/source"
"$build_python" "$source_root/scripts/check_release.py" --root "$source_root" --expected-version "$version"
build_command=(-m build)
if [[ "$no_isolation" -eq 1 ]]; then build_command+=(--no-isolation); fi
"$build_python" "${build_command[@]}" --outdir "$work/artifacts" "$source_root"
"$build_python" -m twine check "$work/artifacts/"*.whl "$work/artifacts/"*.tar.gz
"$build_python" "$source_root/scripts/check_release.py" --root "$source_root" --expected-version "$version" --artifacts "$work/artifacts"
wheels=("$work/artifacts/"*.whl)
sdists=("$work/artifacts/"*.tar.gz)
[[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" && ${#sdists[@]} -eq 1 && -f "${sdists[0]}" ]] || {
  echo 'Expected exactly one wheel and one sdist' >&2; exit 2;
}

# Resolve runtime dependencies normally; optional test tools stay in the tooling venv.
"$build_python" -m pip install "${pip_options[@]}" "${wheels[0]}"
cd "$source_root"
export JARVIS_OPERAS_CURVE_CACHE_ROOT="$work/test-cache"
export JARVIS_OPERAS_PERSIST_FILE="$work/test-state/persist.json"
# Explicit store overrides would defeat tests that select their own legacy base.
unset JARVIS_OPERAS_SOURCES_FILE JARVIS_OPERAS_OVERRIDES_FILE JARVIS_OPERAS_USER_SOURCE_DIR JARVIS_OPERAS_CURVE_INDEX
"$build_python" -m pytest -q tests/test_release_gate_120.py
"$build_python" -m pytest -q
"$build_python" scripts/generate_function_docs.py --out "$work/function-docs"

# Complete installed-wheel validation uses a second venv without system packages.
"$python_bin" -m venv "$work/install"
install_python="$work/install/bin/python"
"$install_python" -m pip install "${pip_options[@]}" "${wheels[0]}"
"$install_python" -m pip check
mkdir "$work/verify"
cd "$work/verify"
"$install_python" -I "$source_root/scripts/smoke_install.py" "$version"
"$install_python" - "$work/artifacts" "$source_root" "$release_root" "$work/source-snapshot.sha256" <<'PY'
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
artifacts, source, repository, snapshot_digest = map(Path, sys.argv[1:])
distributions = sorted(artifacts.glob('*.whl')) + sorted(artifacts.glob('*.tar.gz'))
checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in distributions}
report = {
    'distribution': 'Jarvis-Operas', 'version': metadata.version('Jarvis-Operas'),
    'installation': 'complete wheel with all runtime dependencies in a clean venv',
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'source_snapshot_sha256': snapshot_digest.read_text(encoding='utf-8').strip(),
    'checks': {name: 'passed' for name in ('source_metadata', 'artifact_content', 'twine', 'release_gate', 'full_tests', 'function_docs', 'pip_check', 'installed_cli_unit_smoke')},
    'artifacts': checksums,
    'installed_dependencies': {dist.metadata['Name']: dist.version for dist in metadata.distributions()},
}
(artifacts / 'release-verification.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
(artifacts / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in checksums.items()), encoding='utf-8')
PY
mkdir -p "$output"
# Recheck after potentially long verification to avoid overwriting another run.
[[ -z "$(ls -A "$output")" ]] || { echo "Output was populated during verification: $output" >&2; exit 2; }
cp "$work/artifacts/"* "$output/"
printf '[jorel] Verified complete artifacts: %s\n' "$output"
if [[ "$dry" -eq 0 ]]; then
  "$build_python" -m twine upload --non-interactive -r pypi "$output/$(basename "${wheels[0]}")" "$output/$(basename "${sdists[0]}")"
else
  printf '[jorel] Dry run complete. Verified candidate retained without publication.\n'
fi
