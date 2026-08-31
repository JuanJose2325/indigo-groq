#!/usr/bin/env bash
# Corre las tres tandas de pruebas. Sin dependencias: solo python3.
set -u
cd "$(dirname "$0")"
fallos=0
for prueba in test_*.py; do
    echo "=== $prueba ==="
    python3 "$prueba" || fallos=$((fallos + 1))
    echo
done
if [ "$fallos" -ne 0 ]; then
    echo "$fallos tandas con fallos"
    exit 1
fi
echo "Todo en verde."
