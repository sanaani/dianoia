#!/bin/bash
# The GPU workflow as one tool (10/04/2026). Each step was done by hand before and went wrong at least once: an upload
# chain was backgrounded so its files never arrived, and the launch script lived in a session scratchpad.
#
#   scripts/aws.sh start [hours] [spot|od]   launch g5.xlarge with a self-terminate cap; wait for SSH; install
#                                            packages; record the instance in ~/dianoia-data/work/aws-instance
#   scripts/aws.sh upload <file|dir>...      copy into ~/work at the same relative path (relative to work/ or
#                                            to scripts/), then verify every file's sha256 on the instance
#   scripts/aws.sh run <script.sh>           upload a job script, run its "# smoke:" commands, start it detached
#                                            (refused while another run-*.sh is running); confirm it is running
#   scripts/aws.sh kill <script.sh>          stop a job and every process it started (its whole session)
#   scripts/aws.sh wait <done-file> <name>   block until ~/work/logs/<done-file> exists (prints DONE if it holds
#                                            ALLDONE, else FAILED) or no process matching <name> remains (DIED).
#                                            Job scripts write ALLDONE only when every step succeeded.
#   scripts/aws.sh ssh '<command>'           run a command in ~/work
#   scripts/aws.sh sync-src                  copy src/dianoia to ~/repo/src on the instance, verified
#   scripts/aws.sh extend <hours>            reset the self-terminate cap to <hours> from now
#   scripts/aws.sh download <sums>...        download and verify the files listed in sums files; keep running
#   scripts/aws.sh finish <sums>...          = aws-finish.sh: download, verify, then terminate
#   scripts/aws.sh image                     save the instance (minus ~/work) as the image start uses; after downloads
set -u
# The whole script is one block, parsed before it runs: bash reads a plain script as it goes, so editing it under a
# running watcher made that watcher fail with a syntax error (10/04/2026).
{
WORK=${DIANOIA_DATA:-$HOME/dianoia-data}/work
STATE=${AWS_STATE:-$WORK/aws-instance}  # AWS_STATE: a second instance (e.g. an image smoke test) keeps its own record
# Prebuilt image (item 5, 10/04/2026): the DLAMI plus PKGS, vLLM (~/vllm-env, with ninja) and the Qwen/SONAR model
# caches, with ~/work removed. Made by "image"; its id is kept in $AMI_FILE. Without it, start uses the plain DLAMI.
AMI_FILE=$WORK/aws-ami
BASE_AMI=ami-012ba162b9cd2729c
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
PKGS="transformers peft sentencepiece protobuf tiktoken accelerate convokit"

# Keepalives (10/06/2026): the P10D5 smoke ssh hung after the remote command had ended (a dead link), and the job
# never started; a dead connection now drops within about 2 minutes, so the step fails visibly instead.
ssh_cmd() { echo "ssh -i $HOME/.ssh/dianoia.pem -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 ubuntu@$1"; }
# A pgrep pattern that cannot match the ssh command line carrying it: "job.sh" -> "[j]ob.sh".
pat() { printf '[%s]%s' "${1:0:1}" "${1:1}"; }
load() { [ -f "$STATE" ] || { echo "no instance recorded in $STATE"; exit 1; }; read -r ID IP < "$STATE"; S=$(ssh_cmd "$IP"); }
# put_big <local file> <path under ~/work>: send a large file in 128 MB parts, each verified on arrival, retried up to 3
# times, and skipped if a previous attempt already delivered it; then join and verify the whole file. Sending 3.6 GB as
# one stream failed after 70 minutes with nothing kept (10/06/2026), so one dropped link cost the whole transfer.
BIG=$((256 * 1024 * 1024)) PART_MB=128
put_big() {
  local src=$1 rel=$2 want n i ps dir t0
  want=$(sha256sum "$src" | cut -d' ' -f1)
  [ "$($S "cd ~/work && sha256sum '$rel' 2>/dev/null" 2>/dev/null | cut -d' ' -f1)" = "$want" ] && { echo "already there: $rel"; return 0; }
  n=$(( ($(stat -c %s "$src") + PART_MB * 1048576 - 1) / (PART_MB * 1048576) ))
  dir=.parts/$(printf '%s' "$rel" | tr / _)
  $S "mkdir -p ~/work/$dir" 2>/dev/null || return 1
  for ((i = 0; i < n; i++)); do
    ps=$(dd if="$src" bs=1M skip=$((i * PART_MB)) count=$PART_MB status=none | sha256sum | cut -d' ' -f1)
    for try in 1 2 3; do
      [ "$($S "sha256sum ~/work/$dir/$i 2>/dev/null" 2>/dev/null | cut -d' ' -f1)" = "$ps" ] && break
      t0=$(date +%s)
      dd if="$src" bs=1M skip=$((i * PART_MB)) count=$PART_MB status=none | $S "cat > ~/work/$dir/$i" 2>/dev/null
      echo "  $rel part $((i + 1))/$n try $try: $(( $(date +%s) - t0 ))s"
    done
    [ "$($S "sha256sum ~/work/$dir/$i 2>/dev/null" 2>/dev/null | cut -d' ' -f1)" = "$ps" ] || { echo "part $i of $rel did not arrive"; return 1; }
  done
  $S "cd ~/work && mkdir -p \$(dirname '$rel') && for ((i = 0; i < $n; i++)); do cat $dir/\$i; done > '$rel.joining' && [ \"\$(sha256sum < '$rel.joining' | cut -d' ' -f1)\" = $want ] && mv -f '$rel.joining' '$rel' && rm -rf $dir" 2>/dev/null \
    && echo "upload verified: $rel (whole file)" || { echo "JOIN DID NOT VERIFY: $rel"; return 1; }
}

case "${1:-}" in
start)
  HOURS=${2:-4}; MARKET=${3:-spot}
  AMI=$BASE_AMI; [ -f "$AMI_FILE" ] && AMI=$(cat "$AMI_FILE")
  # SSH is open only to the laptop's public address, which changes (10/08/2026: two launches never connected, and the
  # upload failed silently after a 7-minute wait). Point the rule at today's address before launching.
  SG=sg-0d9d15e3870f9acfc; MYIP=$(curl -s --max-time 10 https://checkip.amazonaws.com | tr -d '[:space:]')
  [[ "$MYIP" =~ ^[0-9.]+$ ]] || { echo "could not read this laptop's public address"; exit 1; }
  for c in $(aws ec2 describe-security-groups --region us-east-1 --group-ids $SG --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text); do
    [ "$c" = "$MYIP/32" ] || { aws ec2 revoke-security-group-ingress --region us-east-1 --group-id $SG --protocol tcp --port 22 --cidr "$c" > /dev/null && echo "ssh rule: removed $c"; }
  done
  aws ec2 describe-security-groups --region us-east-1 --group-ids $SG --query 'SecurityGroups[0].IpPermissions[?FromPort==`22`].IpRanges[].CidrIp' --output text | grep -qxF "$MYIP/32" \
    || { aws ec2 authorize-security-group-ingress --region us-east-1 --group-id $SG --protocol tcp --port 22 --cidr "$MYIP/32" > /dev/null && echo "ssh rule: opened to $MYIP/32"; }
  # Fractional hours (2.5) broke $((HOURS*60)) and launched an instance with no self-terminate cap (10/09/2026).
  MIN=$(awk -v h="$HOURS" 'BEGIN{print int(h*60)}'); [ "$MIN" -gt 0 ] 2>/dev/null || { echo "bad hours: $HOURS"; exit 1; }
  UD=$(printf '#!/bin/bash\nshutdown -h +%d\n' "$MIN" | base64 -w0)
  SPOT=(); [ "$MARKET" = spot ] && SPOT=(--instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}')
  ID=$(MSYS_NO_PATHCONV=1 aws ec2 run-instances --region us-east-1 --image-id "$AMI" \
    --instance-type "${TYPE:-g5.xlarge}" --key-name dianoia "${SPOT[@]}" \
    --instance-initiated-shutdown-behavior terminate \
    --network-interfaces '[{"DeviceIndex":0,"AssociatePublicIpAddress":true,"Groups":["sg-0d9d15e3870f9acfc"]}]' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --user-data "$UD" --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dianoia}]' \
    --query "Instances[0].InstanceId" --output text) || { echo "launch failed (no $MARKET capacity? try: start $HOURS od)"; exit 1; }
  aws ec2 wait instance-running --instance-ids "$ID"
  IP=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "$ID $IP" > "$STATE"; S=$(ssh_cmd "$IP")
  for _ in $(seq 1 40); do $S true 2>/dev/null && break; sleep 10; done
  $S true 2>/dev/null || { echo "SSH NEVER CONNECTED to $IP"; exit 1; }
  INSTALL="sudo /opt/pytorch/bin/pip install -q $PKGS 2>&1 | grep -v notice | tail -1;"; [ "$AMI" != "$BASE_AMI" ] && INSTALL=""
  $S "$INSTALL mkdir -p ~/work/logs; /opt/pytorch/bin/python -c 'import torch, transformers, peft; print(\"ready\", torch.cuda.is_available())'" 2>/dev/null
  echo "$ID $IP (self-terminates in ${HOURS}h, image $AMI)" ;;
image)
  # Save the running instance as the prebuilt image, without ~/work: data travels only by checksummed upload. An EBS
  # snapshot is point-in-time from create-image, so ~/work is moved aside (after sync) only for that call and then
  # restored; the first version deleted it, which broke the next job on the same instance (10/04/2026).
  load
  # Guard (as aws-finish): every result the instance lists in a logs/sums-*.txt must already be here and verify.
  for SUMS in $($S "cd ~/work && ls logs/sums-*.txt 2>/dev/null" 2>/dev/null); do
    (cd "$WORK" && [ -f "$SUMS" ] && sha256sum -c --quiet "$SUMS" >/dev/null 2>&1)       || { echo "NOT DOWNLOADED: $SUMS (run: aws.sh download $SUMS)"; exit 1; }
  done
  $S "mv ~/work ~/.work-hold && mkdir -p ~/work/logs && sync" 2>/dev/null || { echo "could not move ~/work aside"; exit 1; }
  NEW=$(aws ec2 create-image --instance-id "$ID" --no-reboot --name "dianoia-$(date +%Y%m%d-%H%M)"     --description "DLAMI + dianoia packages, vLLM, model caches" --query ImageId --output text)
  $S "rm -rf ~/work && mv ~/.work-hold ~/work" 2>/dev/null || echo "WARNING: ~/work not restored (it is in ~/.work-hold)"
  [ -n "$NEW" ] || exit 1
  # A 50 GB snapshot takes longer than "aws ec2 wait image-available" waits (10 minutes); poll for up to an hour.
  echo "creating $NEW"
  for _ in $(seq 1 60); do
    st=$(aws ec2 describe-images --image-ids "$NEW" --query 'Images[0].State' --output text)
    [ "$st" = available ] && break; [ "$st" = failed ] && { echo "image $NEW failed"; exit 1; }; sleep 60
  done
  [ "$st" = available ] || { echo "image $NEW still $st after an hour; record it later with: echo $NEW > $AMI_FILE"; exit 1; }
  OLD=$([ -f "$AMI_FILE" ] && cat "$AMI_FILE"); echo "$NEW" > "$AMI_FILE"; echo "image ready: $NEW"
  # Keep one image: deregister the previous one and delete its snapshots.
  if [ -n "$OLD" ] && [ "$OLD" != "$NEW" ]; then
    SNAPS=$(aws ec2 describe-images --image-ids "$OLD" --query 'Images[0].BlockDeviceMappings[].Ebs.SnapshotId' --output text)
    aws ec2 deregister-image --image-id "$OLD" && for sn in $SNAPS; do aws ec2 delete-snapshot --snapshot-id "$sn"; done
    echo "removed old image $OLD"
  fi ;;
upload)
  load; shift
  T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
  for f in "$@"; do
    if [ -e "$WORK/$f" ]; then src="$WORK/$f"; rel="$f"; elif [ -e "$SCRIPTS/$f" ]; then src="$SCRIPTS/$f"; rel=$(basename "$f"); else echo "not found: $f"; exit 1; fi
    if [ -f "$src" ] && [ "$(stat -c %s "$src")" -gt $BIG ]; then put_big "$src" "$rel" || { echo "UPLOAD FAILED OR DID NOT VERIFY"; exit 1; }; continue; fi
    [ -d "$src" ] && [ -n "$(find "$src" -type f -size +$((BIG / 1048576))M | head -1)" ] && { echo "$f holds a file over $((BIG / 1048576)) MB: upload it by name so it goes in parts"; exit 1; }
    mkdir -p "$T/$(dirname "$rel")"; cp -r "$src" "$T/$rel"
  done
  [ -n "$(find "$T" -type f | head -1)" ] || exit 0  # every file was large and went in parts
  (cd "$T" && find . -type f -name '*.sh' -exec sed -i 's/\r$//' {} + && find . -type f ! -name .sums | sort | xargs sha256sum > .sums)
  # Send only the files the instance lacks or holds a different version of: relaunching J10 re-sent 190 MB of
  # unchanged weights over the laptop's uplink (10/05/2026). The full .sums is still verified below.
  STALE=$($S 'cd ~/work && cat > .sums.want && sha256sum -c .sums.want 2>/dev/null | grep -v ": OK$" | sed "s/: [^:]*$//"' < "$T/.sums" 2>/dev/null) \
    || STALE=$(sed 's/^[0-9a-f]* [ *]//' "$T/.sums")
  (cd "$T" && sed 's/^[0-9a-f]* [ *]//' .sums | while read -r p; do printf '%s\n' "$STALE" | grep -qxF "$p" || rm -f "$p"; done)
  # One foreground pipeline: the remote side reads the whole archive before verifying. It unpacks into .incoming and
  # renames each file into place: tar rewrote lora.pt in place while J10 was loading it, and J10 died reading a
  # half-written file (10/05/2026). A rename leaves a running job's open file intact.
  (cd "$T" && tar cf - . | xz -T0 -3) | $S 'cd ~/work && rm -rf .incoming && mkdir .incoming && xz -dc | tar xf - -C .incoming && (cd .incoming && find . -type f | while read -r f; do mkdir -p "../$(dirname "$f")" && mv -f "$f" "../$f"; done) && rm -rf .incoming && sha256sum -c --quiet .sums && echo "upload verified: $(wc -l < .sums) files"' 2>/dev/null \
    || { echo "UPLOAD FAILED OR DID NOT VERIFY"; exit 1; } ;;
run)
  # A job script names the files it needs on "# needs:" lines; they are uploaded (and verified) with it. Twice on
  # 10/04/2026 a job failed because a file it loads was no longer on the instance.
  # A bare "wait" waits for every background child, including a sampler loop that never ends: P10SP hung an hour after
  # its work was done (10/06/2026). Name the PIDs.
  grep -nE '(^|[;&|{ ]|then|do)wait[[:space:]]*($|[;})&|])' "$WORK/$2" "$SCRIPTS/$2" 2>/dev/null && { echo "REFUSED: bare 'wait' in $2; wait on named PIDs"; exit 1; }
  load; NEEDS=$(grep -h '^# needs:' "$WORK/$2" "$SCRIPTS/$2" 2>/dev/null | sed 's/^# needs://' | tr '
' ' ')
  "$0" upload "$2" $NEEDS || exit 1
  # Where upload put it: the same relative path under work/, or the bare name for a file from scripts/.
  if [ -e "$WORK/$2" ]; then REL=$2; else REL=$(basename "$2"); fi; NAME=$(basename "$2")
  # Clear the done files the job writes: a FAILED left by an earlier attempt made "wait" report a live rerun as failed
  # (10/04/2026).
  DONES=$(grep -o 'logs/done-[A-Za-z0-9_-]*' "$WORK/$2" "$SCRIPTS/$2" 2>/dev/null | sed 's/.*://' | sort -u | tr '
' ' ')
  [ -n "$DONES" ] && $S "cd ~/work && rm -f $DONES" 2>/dev/null
  # One job at a time (10/05/2026): a rerun started while the old G1 still ran, and both trained on one GPU. Stop the
  # old one with "aws.sh kill <script>" first; FORCE=1 overrides on purpose.
  OTHER=$($S "pgrep -af '[.]/run-[A-Za-z0-9_-]*[.]sh'" 2>/dev/null)
  [ -n "$OTHER" ] && [ "${FORCE:-0}" != 1 ] && { echo "REFUSED: a job is already running (aws.sh kill <script>, or FORCE=1):"; echo "$OTHER"; exit 1; }
  # Smoke first (10/05/2026): G1 crashed twice, and D1 once, on code paths no small run had exercised. Each "# smoke:"
  # line is a command run in ~/work before the job; the job does not start unless every one succeeds.
  grep -h '^# smoke:' "$WORK/$2" "$SCRIPTS/$2" 2>/dev/null | sed 's/^# smoke://' | while read -r CMD; do
    echo "smoke: $CMD"
    # < /dev/null: ssh read the rest of the smoke list from stdin, so only the first line ran (10/09/2026).
    $S "cd ~/work && timeout 1800 bash -c $(printf '%q' "$CMD") >> logs/$NAME.smoke 2>&1" < /dev/null 2>/dev/null || { echo "SMOKE FAILED (logs/$NAME.smoke on the instance)"; exit 1; }
  done || exit 1
  $S "cd ~/work && chmod +x $REL && (setsid nohup ./$REL > logs/$NAME.out 2>&1 < /dev/null &)" 2>/dev/null
  sleep 3; $S "pgrep -f '$(pat "$NAME")' > /dev/null" 2>/dev/null && echo "running: $REL" || { echo "NOT RUNNING: $REL (see logs/$NAME.out)"; exit 1; } ;;
kill)
  # Stop a job and everything it started (10/05/2026): a hand-typed "pkill -f run-G1.sh" matched its own ssh command
  # line and killed the connection, and left the job's python running. A job runs in its own session (setsid in
  # "run"), so the whole session is killed, then checked empty.
  load; NAME=$(basename "$2")
  $S "for p in \$(pgrep -f '$(pat "$NAME")'); do kill -TERM -- -\$p 2>/dev/null; done; sleep 5; for p in \$(pgrep -f '$(pat "$NAME")'); do kill -KILL -- -\$p 2>/dev/null; done; sleep 1; pgrep -af '$(pat "$NAME")' && echo 'STILL RUNNING' || echo 'stopped: $NAME'" 2>/dev/null ;;
wait)
  load
  # Job contract: a job writes ALLDONE to its done file only when every step succeeded, FAILED otherwise.
  # A spot instance can be reclaimed mid-job (10/05/2026): ssh then fails for good, and this looped forever. After 5
  # failed connections in a row, ask EC2; a terminated or stopped instance ends the wait with TERMINATED.
  fails=0
  while :; do
    r=$($S "cd ~/work; if [ -f logs/$2 ]; then grep -q ALLDONE logs/$2 && echo DONE || echo FAILED; elif ! pgrep -f '$(pat "$3")' >/dev/null; then echo DIED; fi" 2>/dev/null)
    rc=$?
    [ -n "$r" ] && break
    if [ $rc -eq 255 ]; then fails=$((fails + 1)); else fails=0; fi  # 255: ssh could not connect
    if [ $fails -ge 5 ]; then
      st=$(aws ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
      case "$st" in terminated|shutting-down|stopped|stopping) echo "TERMINATED ($st)"; exit 1 ;; esac
      fails=0
    fi
    sleep 60
  done
  echo "$r" ;;
sync-src)
  # The dianoia package (src/dianoia) to ~/repo/src on the instance, verified: jobs that import it set
  # PYTHONPATH=/home/ubuntu/repo/src. The image does not carry it (code changes more often than images).
  load; R=$(cd "$SCRIPTS/.." && pwd)
  (cd "$R" && find src/dianoia -name '*.py' | sort | xargs sha256sum > /tmp/dianoia-src.sums && tar cf - src/dianoia /tmp/dianoia-src.sums 2>/dev/null)     | $S 'mkdir -p ~/repo && cd ~/repo && tar xf - && cp tmp/dianoia-src.sums . && sha256sum -c --quiet dianoia-src.sums && echo "src verified: $(wc -l < dianoia-src.sums) files"' 2>/dev/null     || { echo "SRC SYNC FAILED"; exit 1; } ;;
ssh)
  load; $S "cd ~/work && $2" 2>/dev/null ;;
extend)
  # Reset the self-terminate cap to <hours> from now (the instance always keeps a cap).
  load; $S "sudo shutdown -c 2>/dev/null; sudo shutdown -h +$(( ${2:-2} * 60 )) 2>&1 | tail -1" 2>/dev/null
  echo "self-terminate reset to ${2:-2}h from now" ;;
download)
  # Download every file listed in the given sums files (relative to ~/work) and verify; the instance keeps running.
  load; shift; cd "$WORK" || exit 1
  for SUMS in "$@"; do
    # One line: a newline inside the remote command would end tar's file list (lost a file on 10/04).
    FILES=$($S "cd ~/work && awk '{print \$2}' $SUMS" 2>/dev/null | tr '\n' ' '); [ -n "$FILES" ] || { echo "cannot read $SUMS"; exit 1; }
    $S "cd ~/work && tar cf - $SUMS $FILES | xz -T0 -6" 2>/dev/null | xz -dc | tar xf - && sha256sum -c --quiet "$SUMS" && echo "verified: $SUMS"       || { echo "DOWNLOAD DID NOT VERIFY: $SUMS"; exit 1; }
  done ;;
finish)
  load; shift; "$SCRIPTS/aws-finish.sh" "$ID" "$IP" "$WORK" "$@" && rm -f "$STATE" ;;
*)
  sed -n '2,/^set -u/{/^set -u/!p}' "$0"; exit 1 ;;
esac
exit
}
