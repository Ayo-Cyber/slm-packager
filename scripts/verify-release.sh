#!/usr/bin/env bash
# End-to-end release verification: everything a unit test cannot cover.
#
# Usage, from the repo root:   bash scripts/verify-release.sh
#
# Needs ./venv with the engines installed (pip install -e ".[all,dev]") and at
# least tinyllama pulled (slm pull tinyllama). Builds a throwaway venv to prove
# the core-only install path still works.

set -uo pipefail

REPO="$(pwd)"
PY="$REPO/venv/bin/python"
SLM="$REPO/venv/bin/slm"
PORT=8971
LEAN=/tmp/slm-verify-lean
DIST=/tmp/slm-verify-dist
PASS=0
FAIL=0

c_g=$'\033[32m'; c_r=$'\033[31m'; c_b=$'\033[1m'; c_0=$'\033[0m'

ok()   { PASS=$((PASS+1)); echo "${c_g}  PASS${c_0}  $1"; }
bad()  { FAIL=$((FAIL+1)); echo "${c_r}  FAIL${c_0}  $1"; }
step() { echo; echo "${c_b}── $1${c_0}"; }
check() { if [ "$1" = "0" ]; then ok "$2"; else bad "$2"; fi; }

cleanup() {
  [ -n "${SERVER_PID:-}" ] && kill -9 "$SERVER_PID" 2>/dev/null
  rm -rf "$LEAN" "$DIST" /tmp/slm-verify-*.yaml /tmp/slm-verify-*.json 2>/dev/null
  return 0
}
trap cleanup EXIT

echo "${c_b}slm-packager v0.3.0 verification${c_0}"
echo "repo: $REPO"

# ── 1. Static checks ─────────────────────────────────────────────────────────
step "1. Tests, lint, docs, build"

"$PY" -m pytest tests/ -q >/tmp/slm-verify-tests.log 2>&1
check $? "test suite passes ($(grep -oE '[0-9]+ passed' /tmp/slm-verify-tests.log | head -1))"

"$PY" -m black --check slm_packager tests scripts -q >/dev/null 2>&1 && \
  "$PY" -m isort --check-only slm_packager tests scripts -q >/dev/null 2>&1
check $? "lint clean (black + isort)"

"$PY" -m mkdocs build --strict --site-dir /tmp/slm-verify-site >/dev/null 2>&1
check $? "docs build with --strict (no broken links)"
rm -rf /tmp/slm-verify-site

"$PY" -m build --outdir "$DIST" >/dev/null 2>&1
check $? "wheel + sdist build"

"$PY" -m twine check "$DIST"/* >/dev/null 2>&1
check $? "twine check passes"

"$PY" -c "
import zipfile, glob, sys
n = zipfile.ZipFile(glob.glob('$DIST/*.whl')[0]).namelist()
sys.exit(0 if 'slm_packager/registry/models.json' in n else 1)"
check $? "wheel ships the model registry"

# ── 2. The headline change: lean install ─────────────────────────────────────
step "2. Lean install (the v0.3.0 headline)"

"$PY" -m venv "$LEAN" >/dev/null 2>&1
START=$(date +%s)
"$LEAN/bin/pip" install -q "$DIST"/*.whl >/dev/null 2>&1
ELAPSED=$(( $(date +%s) - START ))
check $? "core installs (${ELAPSED}s, $(du -sh "$LEAN" 2>/dev/null | cut -f1))"

"$LEAN/bin/pip" list --format=freeze 2>/dev/null | grep -qiE "^(torch|llama-cpp-python|onnxruntime|transformers|numpy)="
if [ $? -ne 0 ]; then ok "no inference engine pulled into core"; else bad "an engine leaked into core"; fi

"$LEAN/bin/slm" list >/dev/null 2>&1
check $? "'slm list' works with no engine installed"

"$LEAN/bin/slm" init --name t --path /tmp/m.gguf --format gguf \
  --runtime llama_cpp -o /tmp/slm-verify-lean.yaml >/dev/null 2>&1
check $? "'slm init' works with no engine installed"

ERR=$("$LEAN/bin/slm" run /tmp/slm-verify-lean.yaml --prompt hi 2>&1 || true)
case "$ERR" in *"slm-packager[gguf]"*) ok "missing engine names the right extra to install";;
  *) bad "missing engine does not name the gguf extra";; esac

# ── 3. CLI with a real model ─────────────────────────────────────────────────
step "3. Real inference (GGUF via llama.cpp)"

OUT=$("$SLM" run tinyllama --prompt "Name one primary color." --no-stream 2>&1)
echo "$OUT" | grep -q "chat template"
check $? "chat template auto-applied (was only done for transformers before)"

echo "$OUT" | tail -1 | grep -qE "[A-Za-z]"
check $? "produced real text output"

# temperature 0 used to crash transformers and sample at 1.0 in ONNX
sed 's/temperature: 0.7/temperature: 0.0/; s/max_tokens: 512/max_tokens: 24/' \
  ~/.slm/configs/tinyllama.yaml > /tmp/slm-verify-temp0.yaml
A=$("$SLM" run /tmp/slm-verify-temp0.yaml --prompt "Count: one two" --no-stream 2>&1 | tail -1)
B=$("$SLM" run /tmp/slm-verify-temp0.yaml --prompt "Count: one two" --no-stream 2>&1 | tail -1)
[ -n "$A" ] && [ "$A" = "$B" ]
check $? "temperature: 0 is deterministic greedy decoding"

# stop sequences: dropped entirely by transformers before
sed 's/temperature: 0.7/temperature: 0.0/; s/max_tokens: 512/max_tokens: 40/; s/stop: \[\]/stop: ["."]/' \
  ~/.slm/configs/tinyllama.yaml > /tmp/slm-verify-stop.yaml
S1=$("$SLM" run /tmp/slm-verify-stop.yaml --prompt "List three colors." --no-stream 2>&1 | tail -1)
S2=$("$SLM" run /tmp/slm-verify-stop.yaml --prompt "List three colors." --stream 2>&1 | tail -1)
[ "${S1#*.}" = "$S1" ]
check $? "stop sequence excluded from non-streaming output"
[ "$S1" = "$S2" ]
check $? "streaming and non-streaming agree with a stop sequence"

# benchmark must refuse to invent numbers
BOUT=$("$SLM" benchmark tinyllama --runs 2 --max-tokens 32 2>&1)
echo "$BOUT" | grep -q "Tokens/sec (median)"
check $? "benchmark reports median tokens/sec over multiple runs"
echo "$BOUT" | grep -q "Tokens Generated"
check $? "benchmark reports how many tokens it actually generated"

# ── 4. API server ────────────────────────────────────────────────────────────
step "4. API server"

# Detach stdout/stderr fully so the server never holds the script's pipe open.
"$SLM" serve --port $PORT >/tmp/slm-verify-serve.log 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 30); do
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done

curl -sf --max-time 5 "http://127.0.0.1:$PORT/health" | grep -q '"ok"'
check $? "/health responds"

CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X POST \
  "http://127.0.0.1:$PORT/generate" -H "Content-Type: application/json" \
  -d '{"prompt":"hi"}')
[ "$CODE" = "400" ]
check $? "/generate before /load returns 400 (the bug the old README hid)"

curl -sf --max-time 120 -X POST "http://127.0.0.1:$PORT/load" \
  -H "Content-Type: application/json" \
  -d "{\"config_path\":\"$HOME/.slm/configs/tinyllama.yaml\"}" | grep -q success
check $? "/load loads a model"

# A failed load must not unload the working model
curl -s -o /dev/null --max-time 10 -X POST "http://127.0.0.1:$PORT/load" \
  -H "Content-Type: application/json" -d '{"config_path":"/tmp/does-not-exist.yaml"}'
CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 90 -X POST \
  "http://127.0.0.1:$PORT/generate" -H "Content-Type: application/json" \
  -d '{"prompt":"hi","params":{"max_tokens":8,"stream":false}}')
[ "$CODE" = "200" ]
check $? "a failed /load leaves the working model serving (was 400)"

# Chat template parity between API and CLI
curl -sf --max-time 90 -X POST "http://127.0.0.1:$PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hi","params":{"max_tokens":8,"stream":false}}' | grep -q '"text"'
check $? "/generate returns text (chat-templated, like the CLI)"

curl -sf --max-time 90 -X POST "http://127.0.0.1:$PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hi","raw":true,"params":{"max_tokens":8,"stream":false}}' | grep -q '"text"'
check $? '"raw": true bypasses the chat template'

# SSE stream
SSE=$(curl -s -N --max-time 90 -X POST "http://127.0.0.1:$PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hi","params":{"max_tokens":8,"stream":true}}' || true)
case "$SSE" in *"data: "*) ok "SSE streaming emits data: frames";;
  *) bad "no SSE data: frames received";; esac
case "$SSE" in *"[DONE]"*) ok "SSE stream terminates with [DONE]";;
  *) bad "SSE stream missing [DONE] terminator";; esac

# Concurrency — this is what used to corrupt shared KV cache state.
# Wait on these PIDs specifically: a bare `wait` would also wait for the
# backgrounded `slm serve` job, which never exits.
CURL_PIDS=()
for i in 1 2 3 4; do
  curl -s --max-time 120 -X POST "http://127.0.0.1:$PORT/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt":"Hello","params":{"max_tokens":16,"stream":false}}' \
    -o "/tmp/slm-verify-c$i.json" -w "%{http_code}\n" &
  CURL_PIDS+=("$!")
done
wait "${CURL_PIDS[@]}"
GOOD=0
for i in 1 2 3 4; do
  grep -q '"text"' "/tmp/slm-verify-c$i.json" 2>/dev/null && GOOD=$((GOOD+1))
done
[ "$GOOD" = "4" ]
check $? "4 concurrent generations all succeed ($GOOD/4, serialized, no corruption)"

# Distinct outputs prove no cross-request contamination of shared model state.
UNIQ=$(cat /tmp/slm-verify-c1.json /tmp/slm-verify-c2.json /tmp/slm-verify-c3.json \
       /tmp/slm-verify-c4.json 2>/dev/null | tr '}' '\n' | sort -u | grep -c '"text"')
[ "$UNIQ" -ge 2 ]
check $? "concurrent outputs are independent ($UNIQ distinct of 4)"

# Client disconnect mid-stream used to wedge the server permanently
curl -s -N --max-time 1 -X POST "http://127.0.0.1:$PORT/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Tell me a long story","params":{"max_tokens":400,"stream":true}}' \
  >/dev/null 2>&1
sleep 2
START=$(date +%s)
curl -sf --max-time 90 -X POST "http://127.0.0.1:$PORT/load" \
  -H "Content-Type: application/json" \
  -d "{\"config_path\":\"$HOME/.slm/configs/tinyllama.yaml\"}" >/dev/null 2>&1
RC=$?
SWITCH=$(( $(date +%s) - START ))
[ "$RC" = "0" ] && [ "$SWITCH" -lt 30 ]
check $? "model switch after a mid-stream disconnect completes (${SWITCH}s; used to hang forever)"

# Shutdown must actually exit
set +m 2>/dev/null
kill "$SERVER_PID" 2>/dev/null
for _ in $(seq 1 15); do
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$SERVER_PID" 2>/dev/null; then
  bad "server did not exit on SIGTERM"
  kill -9 "$SERVER_PID" 2>/dev/null
else
  ok "server shuts down cleanly on SIGTERM (used to hang)"
fi

grep -qi "traceback" /tmp/slm-verify-serve.log
if [ $? -ne 0 ]; then ok "no tracebacks in the server log"; else bad "tracebacks in server log"; fi

# ── 5. Registry ──────────────────────────────────────────────────────────────
step "5. Registry health"
"$PY" scripts/check_registry.py 2>&1 | tail -1 | grep -q "All registry targets resolve"
check $? "all registry models resolve on HuggingFace"

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "${c_b}──────────────────────────────────────${c_0}"
if [ "$FAIL" = "0" ]; then
  echo "${c_g}${c_b}ALL $PASS CHECKS PASSED${c_0}"
else
  echo "${c_r}${c_b}$FAIL failed${c_0}, $PASS passed"
fi
echo "${c_b}──────────────────────────────────────${c_0}"
exit $([ "$FAIL" = "0" ] && echo 0 || echo 1)
