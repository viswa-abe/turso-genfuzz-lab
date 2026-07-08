#!/bin/sh
set -eu

VERSION="v0.7.0-pre.10"
ASSET="turso_cli-x86_64-unknown-linux-gnu.tar.xz"
SHA256="7953bcfb301b3cdd2c8032a7813e2af17b7aa9e917c80b300e861b03a8173ba9"
URL="https://github.com/tursodatabase/turso/releases/download/${VERSION}/${ASSET}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="${ROOT}/.workers/tmp"
VENDOR="${ROOT}/.workers/vendor"
BIN="${VENDOR}/bin"
LIB="${VENDOR}/lib"

mkdir -p "${TMP}" "${BIN}" "${LIB}" "${VENDOR}"

ARCHIVE="${TMP}/${ASSET}"
curl --proto '=https' --tlsv1.2 -fsSL "${URL}" -o "${ARCHIVE}"

printf '%s  %s\n' "${SHA256}" "${ARCHIVE}" | sha256sum -c -
rm -rf "${TMP}/turso_cli-x86_64-unknown-linux-gnu"
tar -C "${TMP}" -xf "${ARCHIVE}"

TURSODB="$(find "${TMP}" -type f -name tursodb -perm -111 | head -n 1)"
if [ -z "${TURSODB}" ]; then
  echo "tursodb binary not found in ${ASSET}" >&2
  exit 1
fi

cp "${TURSODB}" "${BIN}/tursodb"
chmod +x "${BIN}/tursodb"

rm -rf "${LIB}"
mkdir -p "${LIB}"
if command -v ldd >/dev/null 2>&1; then
  ldd "${BIN}/tursodb" |
    awk '{ if ($3 ~ /^\//) print $3; else if ($1 ~ /^\//) print $1 }' |
    while IFS= read -r dep; do
      cp -L "${dep}" "${LIB}/"
    done
fi

if [ "$(uname -s)" = "Linux" ]; then
  if [ -x "${LIB}/ld-linux-x86-64.so.2" ]; then
    "${LIB}/ld-linux-x86-64.so.2" --library-path "${LIB}" "${BIN}/tursodb" --version
  else
    "${BIN}/tursodb" --version
  fi
else
  echo "installed Linux tursodb ${VERSION} at ${BIN}/tursodb"
fi
