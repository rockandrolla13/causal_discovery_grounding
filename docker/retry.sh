#!/bin/sh
set -eu

max_attempts="$1"
shift
attempt=1

until "$@"; do
    if [ "${attempt}" -ge "${max_attempts}" ]; then
        exit 1
    fi
    attempt=$((attempt + 1))
    echo "Retrying command (${attempt}/${max_attempts})" >&2
    sleep 5
done
