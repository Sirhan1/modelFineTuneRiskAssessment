#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

pick_supported_python() {
  local candidate
  for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  # Check typical Homebrew versioned binary locations even when not linked.
  for candidate in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 \
                   /usr/local/bin/python3.12 /usr/local/bin/python3.11 /usr/local/bin/python3.10; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  # Check Homebrew opt prefixes for installed formulas.
  if command -v brew >/dev/null 2>&1; then
    for candidate in python@3.12 python@3.11 python@3.10; do
      local prefix
      prefix="$(brew --prefix "$candidate" 2>/dev/null || true)"
      if [[ -n "$prefix" && -x "$prefix/bin/${candidate/python@/python}" ]]; then
        echo "$prefix/bin/${candidate/python@/python}"
        return 0
      fi
    done
  fi

  return 1
}

PYTHON_BIN="$(pick_supported_python || true)"

if [[ -z "$PYTHON_BIN" && -x /opt/homebrew/bin/brew ]]; then
  echo "No compatible Python found. Installing Homebrew python@3.12..." >&2
  /opt/homebrew/bin/brew install python@3.12
  if [[ -x /opt/homebrew/opt/python@3.12/bin/python3.12 ]]; then
    PYTHON_BIN="/opt/homebrew/opt/python@3.12/bin/python3.12"
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No compatible Python found (need 3.10, 3.11, or 3.12)." >&2
  echo "Install one with: brew install python@3.11" >&2
  exit 1
fi

PY_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VERSION" in
  3.10|3.11|3.12) ;;
  *)
    echo "Detected Python $PY_VERSION from $PYTHON_BIN." >&2
    echo "Use Python 3.10, 3.11, or 3.12 for this project." >&2
    echo "Tip: brew install python@3.11" >&2
    exit 1
    ;;
esac

echo "Using $PYTHON_BIN (Python $PY_VERSION)"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Run tests with: make test"
echo "Run demo with: make demo"
