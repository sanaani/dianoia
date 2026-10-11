#!/bin/bash
# Download every file in a manifest, resumable, and verify each checksum (10/05/2026).
#
#   scripts/fetch-verified.sh <manifest> [dir] [name]
#
# With a name, only that file is fetched: the pipeline (pipelines/) runs one per file, in parallel. Serially, the five
# patent tables took over an hour at about 1 MB/s each (10/05/2026).
# Manifest lines: "<name> <md5:hex|sha256:hex> <size> <url>" ('#' lines skipped). Prints OK, BAD CHECKSUM or FAILED per
# file and exits 1 unless every file is OK. Carriage returns are stripped first: a manifest written by Windows Python
# ended each URL in "\r", and curl rejected all five patent files (10/05/2026).
set -u
MANIFEST=$1; DIR=${2:-$(dirname "$MANIFEST")}; ONLY=${3:-}
bad=0
while read -r name sum size url; do
  [ -z "${name:-}" ] || [ "${name:0:1}" = "#" ] && continue
  [ -n "$ONLY" ] && [ "$name" != "$ONLY" ] && continue
  # A complete file is only verified: resuming one makes the server answer 416 and curl -f fail.
  if [ "$(stat -c %s "$DIR/$name" 2>/dev/null)" = "$size" ]; then :
  elif ! curl -sSfL --retry 5 --retry-delay 10 -C - -o "$DIR/$name" "$url"; then echo "FAILED $name"; bad=1; continue; fi
  algo=${sum%%:*}; want=${sum#*:}
  got=$("${algo}sum" "$DIR/$name" | cut -d' ' -f1)
  if [ "$got" = "$want" ]; then echo "OK $name"; else echo "BAD CHECKSUM $name"; bad=1; fi
done < <(tr -d '\r' < "$MANIFEST")
[ -n "$ONLY" ] && ! tr -d '\r' < "$MANIFEST" | grep -q "^$ONLY " && { echo "NOT IN MANIFEST $ONLY"; bad=1; }
exit $bad
