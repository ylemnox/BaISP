#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Deprecated: use ./run_baisp_gui.sh"
exec "$SCRIPT_DIR/run_baisp_gui.sh"
