#!/usr/bin/env bash
set -euo pipefail

# Config
URL="http://images.cocodataset.org/zips/val2014.zip"
OUT_DIR="LLava/Benchmark/VQAv2/Images/mscoco"
OUT_ZIP="${OUT_DIR}/val2014.zip"
LOG_DIR="${OUT_DIR}/_logs"
LOG_FILE="${LOG_DIR}/val2014_download.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "[Info] Target: ${OUT_ZIP}"
echo "[Info] Log:    ${LOG_FILE}"

# Prefer fastest tool: aria2c -> axel -> wget -> curl
if command -v aria2c >/dev/null 2>&1; then
  echo "[Use] aria2c (multi-connection)"
  nohup aria2c -c -x 16 -s 16 -k 1M --file-allocation=none -o "$(basename "${OUT_ZIP}")" \
    --dir="${OUT_DIR}" "${URL}" >>"${LOG_FILE}" 2>&1 &
  PID=$!
elif command -v axel >/dev/null 2>&1; then
  echo "[Use] axel (multi-connection)"
  nohup axel -a -n 16 -o "${OUT_ZIP}" "${URL}" >>"${LOG_FILE}" 2>&1 &
  PID=$!
elif command -v wget >/dev/null 2>&1; then
  echo "[Use] wget (resume)"
  nohup wget -c --tries=0 --timeout=30 --read-timeout=30 \
    --user-agent="Mozilla/5.0" -O "${OUT_ZIP}" "${URL}" >>"${LOG_FILE}" 2>&1 &
  PID=$!
elif command -v curl >/dev/null 2>&1; then
  echo "[Use] curl (resume)"
  nohup curl -fL -C - -o "${OUT_ZIP}" "${URL}" >>"${LOG_FILE}" 2>&1 &
  PID=$!
else
  echo "[Error] No downloader found (aria2c/axel/wget/curl). Please install one."
  exit 1
fi

echo "[Started] PID=${PID}"
chmod +x "$0" || true

echo "[Tips] 查看进度: tail -f \"${LOG_FILE}\""
echo "[Tips] 文件大小: ls -lh \"${OUT_ZIP}\" || true"
