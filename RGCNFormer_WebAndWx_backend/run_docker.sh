#!/bin/bash
# Thin wrapper — delegates to scripts/deploy/run_docker.sh
exec "$(dirname "$0")/scripts/deploy/run_docker.sh" "$@"
