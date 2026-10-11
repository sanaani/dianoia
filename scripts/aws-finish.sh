#!/bin/bash
# Download a run's outputs and terminate its instance, in that order and only if every file verifies (10/04/2026).
# On 10/04 an instance was terminated before an adapter (M1) was downloaded; this makes that impossible.
#
#   scripts/aws-finish.sh <instance-id> <ip> <local-work-dir> <remote sums file> [<remote sums file> ...]
#
# Each sums file (relative to ~/work on the instance) lists "sha256  path" lines. Every listed file is downloaded
# into <local-work-dir> at the same relative path and checked with sha256sum -c. The instance is terminated only if
# every check passes; otherwise nothing is terminated and the exit code is 1.
#
# Unlisted outputs (10/06/2026): the P10D2 instance was terminated with its trained model on it, because the job's sums
# listed only the scores, and 36 GPU minutes had to be redone. So before terminating, every file over 1 MB under ~/work
# on the instance must also be here at the same path and size (inputs were uploaded from here, outputs just came down),
# or start with a prefix listed in a ~/work/logs/discard-*.txt the job wrote, or sit under a *-smoke* path. Anything
# else refuses the terminate and is printed. A job that means to leave a file behind says so in its discard file.
set -u
ID=$1; IP=$2; DEST=$3; shift 3
S="ssh -i $HOME/.ssh/dianoia.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 ubuntu@$IP"
cd "$DEST" || exit 1
for SUMS in "$@"; do
  # One line: a newline inside the remote command would end tar's file list.
  FILES=$($S "cd ~/work && awk '{print \$2}' $SUMS" 2>/dev/null | tr '\n' ' ') || { echo "cannot read $SUMS"; exit 1; }
  [ -n "$FILES" ] || { echo "$SUMS lists no files"; exit 1; }
  $S "cd ~/work && tar cf - $SUMS $FILES | xz -T0 -6" 2>/dev/null | xz -dc | tar xf - || { echo "download failed for $SUMS"; exit 1; }
  sha256sum -c "$SUMS" || { echo "CHECKSUM FAILED: $SUMS; instance left running"; exit 1; }
done
DISCARD=$($S "cat ~/work/logs/discard-*.txt 2>/dev/null" 2>/dev/null)
LEFT=$($S "cd ~/work && find . -type f -size +1M -not -path '*-smoke*' -printf '%s %P\n'" 2>/dev/null) || { echo "cannot list ~/work"; exit 1; }
MISSING=""
while read -r SIZE F; do
  [ -n "$F" ] || continue
  [ -f "$F" ] && [ "$(stat -c %s "$F")" = "$SIZE" ] && continue
  KEEP=1
  while read -r P; do [ -n "$P" ] && [ "${F#"$P"}" != "$F" ] && KEEP=0 && break; done <<< "$DISCARD"
  [ $KEEP = 1 ] && MISSING="$MISSING  $SIZE $F
"
done <<< "$LEFT"
if [ -n "$MISSING" ]; then
  echo "NOT DOWNLOADED (instance left running; add to a sums file or to logs/discard-<job>.txt):"
  printf '%s' "$MISSING"
  exit 1
fi
echo "all downloads verified; terminating $ID"
aws ec2 terminate-instances --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text
