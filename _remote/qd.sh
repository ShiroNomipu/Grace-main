#!/bin/bash
# Resident queue (approved 2026-10-17): keeps the GPU busy; never shuts the instance down
# unless runs/AUTO_SHUTDOWN exists and nothing is pending or running.
#
# Jobs: runs/qd/*.jobs, one per line: RUN_ID|script|args|label[|needs]
#   taken in file-name order, then line order; files may be added or appended at any time.
#   needs: comma-separated RUN_IDs that must end with exit code 0 first. If one of them
#   failed, the job is recorded as skipped (exit_code "dep_failed") and never run.
# Slots: training jobs run while (training processes on the box) < $(cat runs/7b3_slots).
#   eval_dump_preds.py jobs run while (eval processes) < $(cat runs/qd_eval_slots, default 1)
#   or while the total is below the training slots; the total never exceeds slots + eval_slots.
#   2026-10-17: runs/qd_eval_slots = 0 until the 3-training memory peak is measured. Processes are counted over the whole box,
#   so the queue coexists with queue_7b3.sh once that queue has launched all of its jobs.
# Guards: runs/qd_PAUSE -> launch nothing new; runs/qd_STOP -> exit (running jobs continue).
#   Two consecutive failed runs create runs/qd_PAUSE. No launch with < 4 GB free on the data disk.
# Summary: when every run of runs/qd/<name>.jobs has an exit code, runs/<name>_summary.txt and
#   runs/<name>_DONE are written.
# Run dirs are never reused: an existing outputs/runs/<RUN_ID> means "already taken".
set -u
ROOT=${QD_ROOT:-/root/autodl-tmp/grace}
cd "$ROOT" || exit 1
if [ -z "${QD_TEST:-}" ]; then source _remote/env_grace.sh >/dev/null 2>&1; cd "$ROOT" || exit 1; fi
export GRACE_MODEL_PATH=$ROOT/models/Qwen2.5-7B-Instruct
export PYTHONPATH=$ROOT
PY=${QD_PY:-python3}
POLL=${QD_POLL:-60}
STAGGER=${QD_STAGGER:-180}
MACH="AutoDL RTX 6000D 85GB / torch 2.9.0+cu128 / transformers 4.56.1"
QLOG=runs/qd.log
mkdir -p runs/qd outputs/runs

log() { echo "[$(date '+%F %T')] $*" >> "$QLOG"; }
reg() { echo "$1" >> runs/registry.jsonl; }
slots() { cat runs/7b3_slots 2>/dev/null || echo 3; }
eval_slots() { cat runs/qd_eval_slots 2>/dev/null || echo 1; }
free_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9; }
n_train() { pgrep -fc -- "python3? (-u )?(run_v1\.py|run_pilot3_dialogue_visual\.py|_remote/smoke_)"; }
n_eval() { pgrep -fc -- "python3? (-u )?eval_dump_preds\.py"; }

run_one() {
  local RID=$1 SCRIPT=$2 ARGS=$3 LABEL=$4 BATCH=$5
  local D=outputs/runs/$RID
  echo "$ARGS" > "$D/args.txt"; echo "$SCRIPT" > "$D/script.txt"
  echo "$GRACE_MODEL_PATH" > "$D/model_path.txt"; echo "$LABEL" > "$D/label.txt"
  reg "{\"run_id\": \"$RID\", \"card\": \"qd $BATCH\", \"status\": \"running\", \"date\": \"$(date +%FT%T)\", \"machine\": \"$MACH\", \"backbone\": \"Qwen2.5-7B-Instruct\", \"script\": \"$SCRIPT\", \"args\": \"$ARGS\", \"label\": \"$LABEL\"}"
  log "$RID: start [$BATCH | $LABEL] $SCRIPT $ARGS"
  # shellcheck disable=SC2086
  GRACE_OUTPUT_DIR=$PWD/$D $PY "$SCRIPT" $ARGS > "$D/train.log" 2>&1
  local RC=$?
  echo $RC > "$D/exit_code"
  local ST=done; [ $RC -ne 0 ] && ST=failed
  reg "{\"run_id\": \"$RID\", \"status\": \"$ST\", \"date\": \"$(date +%FT%T)\", \"exit_code\": $RC}"
  log "$RID: end rc=$RC"
  if [ $RC -ne 0 ]; then
    echo "$RID" >> runs/qd_fail_streak
    if [ "$(wc -l < runs/qd_fail_streak)" -ge 2 ]; then
      touch runs/qd_PAUSE; log "PAUSE: 2 consecutive failures ($(tr '\n' ' ' < runs/qd_fail_streak)); remove runs/qd_PAUSE to resume"
    fi
  else
    : > runs/qd_fail_streak
  fi
}

# one pass over the job files; launches at most one job per call (then the caller staggers)
try_launch() {
  local F B RID SCRIPT ARGS LABEL NEEDS N E st full_t=0 full_e=0
  for F in $(ls runs/qd/*.jobs 2>/dev/null | sort); do
    B=$(basename "$F" .jobs)
    while IFS='|' read -r RID SCRIPT ARGS LABEL NEEDS; do
      [ -z "$RID" ] && continue
      case "$RID" in \#*) continue;; esac
      [ -e "outputs/runs/$RID" ] && continue
      st=ready
      for N in ${NEEDS//,/ }; do
        E=outputs/runs/$N/exit_code
        if [ ! -e "$E" ]; then st=wait; break; fi
        if [ "$(cat "$E")" != 0 ]; then st=skip; break; fi
      done
      if [ $st = skip ]; then
        mkdir -p "outputs/runs/$RID"; echo dep_failed > "outputs/runs/$RID/exit_code"
        reg "{\"run_id\": \"$RID\", \"status\": \"skipped\", \"date\": \"$(date +%FT%T)\", \"reason\": \"dependency $N did not end with exit code 0\"}"
        log "$RID: skipped, dependency $N failed"; continue
      fi
      [ $st = wait ] && continue
      # keep file order within each kind: once one job of a kind finds no slot, later ones wait too
      if [ "$SCRIPT" = eval_dump_preds.py ]; then
        [ $full_e = 1 ] && continue
        [ "$(n_eval)" -lt "$(eval_slots)" ] || [ $(( $(n_train) + $(n_eval) )) -lt "$(slots)" ] || { full_e=1; continue; }
      else
        [ $full_t = 1 ] && continue
        [ "$(n_train)" -lt "$(slots)" ] && [ $(( $(n_train) + $(n_eval) )) -lt $(( $(slots) + $(eval_slots) )) ] || { full_t=1; continue; }
      fi
      if [ "$(free_gb)" -lt 4 ]; then
        [ -e runs/qd_lowdisk ] || { touch runs/qd_lowdisk; log "LOW DISK: $(free_gb)G free < 4G, not launching"; }
        return 1
      fi
      rm -f runs/qd_lowdisk
      mkdir "outputs/runs/$RID" || continue
      run_one "$RID" "$SCRIPT" "$ARGS" "$LABEL" "$B" < /dev/null &
      return 0
    done < "$F"
  done
  return 1
}

summarise_done() {
  local F B RID SCRIPT REST all
  for F in $(ls runs/qd/*.jobs 2>/dev/null | sort); do
    B=$(basename "$F" .jobs)
    [ -e "runs/${B}_DONE" ] && continue
    all=1
    while IFS='|' read -r RID SCRIPT REST; do
      [ -z "$RID" ] && continue
      case "$RID" in \#*) continue;; esac
      [ -e "outputs/runs/$RID/exit_code" ] || { all=0; break; }
    done < "$F"
    [ $all = 1 ] || continue
    if grep -v '^#' "$F" | grep -q -v '|eval_dump_preds.py|'; then
      $PY runs/summarise_queue.py "$F" > "runs/${B}_summary.txt" 2>&1
    else
      grep -v '^#' "$F" | while IFS='|' read -r RID SCRIPT REST; do
        [ -n "$RID" ] && echo "$RID rc=$(cat outputs/runs/$RID/exit_code) ${REST%%|*}"
      done > "runs/${B}_summary.txt"
    fi
    touch "runs/${B}_DONE"; log "batch $B finished; summary runs/${B}_summary.txt"
  done
}

pending() {
  local F RID REST
  for F in runs/qd/*.jobs; do
    [ -e "$F" ] || continue
    while IFS='|' read -r RID REST; do
      [ -z "$RID" ] && continue
      case "$RID" in \#*) continue;; esac
      [ -e "outputs/runs/$RID" ] || { echo 1; return; }
    done < "$F"
  done
  echo 0
}

log "qd start pid=$$ slots=$(slots) eval_slots=$(eval_slots) free=$(free_gb)G autoshutdown=$([ -e runs/AUTO_SHUTDOWN ] && echo on || echo off)"
echo $$ > runs/qd.pid
while true; do
  date +%FT%T > runs/qd_heartbeat
  if [ -e runs/qd_STOP ]; then log "qd_STOP found, exiting (running jobs continue)"; break; fi
  if [ ! -e runs/qd_PAUSE ]; then
    while try_launch; do sleep "$STAGGER"; done
  fi
  summarise_done
  if [ -e runs/AUTO_SHUTDOWN ] && [ "$(pending)" = 0 ] && [ $(( $(n_train) + $(n_eval) )) -eq 0 ]; then
    log "queue empty, nothing running, runs/AUTO_SHUTDOWN present: shutting down in 120 s"
    sleep 120; sync; [ -z "${QD_TEST:-}" ] && shutdown; break
  fi
  sleep "$POLL"
done
