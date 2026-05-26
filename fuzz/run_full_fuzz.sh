#!/usr/bin/env bash
# Full deep fuzz campaign across every ordvec cargo-fuzz target.
#
# Tuned for a high-core workstation (e.g. Ryzen 9950X / 128 GB DDR5). Runs each
# target in libFuzzer *fork* mode across all cores for a per-target wall-clock
# budget, persisting the corpus (runs are cumulative / resumable) and collecting
# any crash artifacts. Fork mode is resilient: a crashing/OOMing child records
# its artifact and the campaign keeps going, so one bad input never ends the run.
#
# Requires: a nightly toolchain and cargo-fuzz (`cargo install cargo-fuzz`).
#
# HEAVY BY DEFAULT. The defaults are a long, many-core campaign (~3h x 7
# targets ~= 21h total; FORKS = cores - 2; peak RAM ~= FORKS x RSS_LIMIT_MB)
# tuned for a big workstation. On a laptop or smaller box, DIAL IT DOWN with the
# env knobs below so you don't peg every core or exhaust RAM. A quick run:
#
#   SECS_PER_TARGET=120 FORKS=2 ./fuzz/run_full_fuzz.sh   # ~14 min on 2 cores
#
# The script prints the estimated total time + RAM up front and, when run
# interactively, waits 5s so you can Ctrl-C and re-run with smaller knobs.
#
# Launch it detached so it survives the terminal/session:
#
#   setsid nohup ./fuzz/run_full_fuzz.sh > fuzz/full_fuzz_run.log 2>&1 &
#   tail -f fuzz/full_fuzz_run.log        # watch
#   pkill -f 'cargo.*fuzz run'            # stop early (corpus is kept)
#
# Tunables (env):
#   SECS_PER_TARGET   per-target wall-clock budget   (default 10800 = 3h)
#   FORKS             concurrent fork workers        (default = nproc - 2)
#   RSS_LIMIT_MB      per-process RSS cap            (default 3072)
#   TARGETS           space-separated target list    (default = all seven)
#
# Examples:
#   SECS_PER_TARGET=43200 ./fuzz/run_full_fuzz.sh          # 12h per target
#   TARGETS="fastscan_b2 search_rankquant" ./fuzz/run_full_fuzz.sh
set -u

cd "$(dirname "$0")/.." || exit 1

# Portable UTC timestamp — GNU `date -Is` / `-I` isn't available on BSD/macOS.
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Ctrl-C stops the whole campaign, not just the current target's fuzzer —
# without this, killing one `cargo fuzz` lets the loop march on to the next.
trap 'echo; echo "interrupted — stopping campaign (corpus kept)."; exit 130' INT

SECS_PER_TARGET="${SECS_PER_TARGET:-10800}"
# Default to all cores but two, so the machine stays responsive and a small box
# isn't pegged; override with FORKS=<n>. Peak RAM is roughly FORKS x
# RSS_LIMIT_MB. CPU count is detected portably (Linux nproc, then getconf, then
# BSD/macOS sysctl, else 1) so `set -u` never sees an empty NCPU.
NCPU="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
FORKS="${FORKS:-$(( NCPU > 2 ? NCPU - 2 : 1 ))}"
RSS_LIMIT_MB="${RSS_LIMIT_MB:-3072}"
TARGETS="${TARGETS:-load_rank load_rankquant load_bitmap load_sign_bitmap roundtrip_rankquant search_rankquant fastscan_b2}"

read -ra _targets <<<"${TARGETS}"
n_targets=${#_targets[@]}
total_secs=$(( SECS_PER_TARGET * n_targets ))
echo "=== ordvec full fuzz campaign ==="
echo "start:          $(now)"
echo "secs/target:    ${SECS_PER_TARGET}  (~$(( SECS_PER_TARGET / 60 ))m each)"
echo "targets:        ${n_targets} — ${TARGETS}"
echo "est. total:     ~$(( total_secs / 3600 ))h $(( total_secs % 3600 / 60 ))m  (targets run sequentially)"
echo "forks:          ${FORKS}  (of ${NCPU} cores)"
echo "rss limit (MB): ${RSS_LIMIT_MB}  → peak RAM ~$(( FORKS * RSS_LIMIT_MB / 1024 )) GB"
echo "host:           $(uname -srm)"
echo
# Interactive abort window: when stdout is a terminal, pause so a heavy run can
# be cancelled and re-launched with smaller knobs. Skipped under redirection
# (e.g. nohup ... > log) so detached campaigns start immediately.
if [ -t 1 ]; then
  echo "Heavy run — Ctrl-C within 5s to abort (or re-run with SECS_PER_TARGET=… FORKS=…)."
  sleep 5
  echo
fi

# Build once up front so a compile error fails fast (not mid-campaign).
echo "=== building all fuzz targets (release) ==="
if ! cargo +nightly fuzz build; then
  echo "FATAL: fuzz build failed; aborting campaign." >&2
  exit 1
fi
echo

mkdir -p fuzz/corpus fuzz/artifacts

any_fail=0
for t in "${_targets[@]}"; do
  echo "############################################################"
  echo "### target: ${t}   started $(now)"
  echo "############################################################"
  mkdir -p "fuzz/corpus/${t}" "fuzz/artifacts/${t}"
  cargo +nightly fuzz run "${t}" -- \
      -fork="${FORKS}" \
      -ignore_crashes=1 \
      -rss_limit_mb="${RSS_LIMIT_MB}" \
      -max_total_time="${SECS_PER_TARGET}" \
      -print_final_stats=1
  rc=$?
  echo "### target ${t} finished $(now) (libfuzzer rc=${rc})"
  if [ "${rc}" -ne 0 ]; then
    echo "### WARNING: ${t} exited non-zero (rc=${rc}) — recorded as a campaign failure."
    any_fail=1
  fi
  echo
done

echo "============================================================"
echo "=== campaign complete $(now) — summary ==="
status=0
crashes=$(find fuzz/artifacts -type f \
  \( -name 'crash-*' -o -name 'oom-*' -o -name 'timeout-*' -o -name 'leak-*' \) 2>/dev/null)
if [ -n "${crashes}" ]; then
  echo "ARTIFACTS FOUND — investigate before publishing:"
  echo "${crashes}"
  status=1
fi
if [ "${any_fail}" -ne 0 ]; then
  echo "One or more fuzz targets exited non-zero (see WARNING lines above)."
  status=1
fi
if [ "${status}" -eq 0 ]; then
  echo "CLEAN: no crash / oom / timeout / leak artifacts, every target exited 0."
fi
echo
echo "corpus sizes:"
du -sh fuzz/corpus/* 2>/dev/null
exit "${status}"
