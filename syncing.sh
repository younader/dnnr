#! /usr/bin/env bash

if [ "$1" == "--help" ]; then
    echo """
This script watches your files and syncs them a remote server.

Usage:

    ./syncing.sh username@server:/path/to/remote/repo

The argument must be a valid scp destination.
    """
    exit 0
fi

TARGET="$1"

echo "$TARGET"

while true; do
  rsync -azv \
      --copy-links \
      --delete \
      --exclude '*.egg-info' \
      --exclude '**pycache**' \
      --exclude '/.mypy_cache/' \
      --exclude '/.pytest_cache/' \
      --exclude '.covera*' \
      --exclude '.coverage*' \
      --exclude 'htmlcov' \
      . "$TARGET"
  date
  sleep 0.5

  if [[ "$(uname)" = "Darwin" ]]; then
      # For Mac OS X
      fswatch --latency 0.5 --one-event --follow-links "`pwd`"
  else
      # hope it is Linux / no support for windows
      inotifywait -r -e modify,create,delete "`pwd`"
  fi
done
