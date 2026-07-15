#!/bin/bash
# Thin wrapper — delegates to scripts/deploy/start_backend.sh
exec "$(dirname "$0")/scripts/deploy/start_backend.sh" "$@"
