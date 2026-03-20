#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [-f]

Create minimal Bazel configuration files for this template repo.
Options:
  -f    Force overwrite of existing files.

This script will create (if missing):
  - WORKSPACE
  - .bazelrc
  - MODULE.bazel
EOF
}

FORCE=0
while getopts ":f" opt; do
  case ${opt} in
    f) FORCE=1 ;;
    *) usage; exit 1 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Initializing minimal Bazel layout in: $ROOT"

# WORKSPACE
if [ -f "$ROOT/WORKSPACE" ] && [ "$FORCE" -ne 1 ]; then
  echo "WORKSPACE exists; skipping"
else
  cat > "$ROOT/WORKSPACE" <<'EOF'
# Minimal Bazel WORKSPACE for template repository
EOF
  echo "Created: $ROOT/WORKSPACE"
fi

# Note: this template intentionally does not create example source files
# or repository BUILD targets. Only Bazel configuration files are created.

# .bazelrc (minimal)
if [ -f "$ROOT/.bazelrc" ] && [ "$FORCE" -ne 1 ]; then
  echo ".bazelrc exists; skipping"
else
  cat > "$ROOT/.bazelrc" <<'EOF'
# Minimal .bazelrc for template
build --verbose_failures
EOF
  echo "Created: $ROOT/.bazelrc"
fi

# MODULE.bazel (minimal module file)
if [ -f "$ROOT/MODULE.bazel" ] && [ "$FORCE" -ne 1 ]; then
  echo "MODULE.bazel exists; skipping"
else
  cat > "$ROOT/MODULE.bazel" <<'EOF'
module(name = "template")
EOF
  echo "Created: $ROOT/MODULE.bazel"
fi

echo "Minimal Bazel layout ready. To run:"
echo "  bash $0"
echo "Examples:"
echo "  bash $0            # create files if missing"
echo "  bash $0 -f         # force overwrite"

exit 0
