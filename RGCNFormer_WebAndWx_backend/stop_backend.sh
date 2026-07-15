#!/bin/bash
# Thin wrapper — delegates to scripts/deploy/stop_backend.sh
exec "$(dirname "$0")/scripts/deploy/stop_backend.sh" "$@"
