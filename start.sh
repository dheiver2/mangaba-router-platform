#!/usr/bin/env bash
# Inicia a API Gemma 4 usando o Python portátil do próprio HD externo.
# 100% autocontido — não depende de nada instalado no notebook.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/python/bin/python3.11"

if [ ! -x "$PYTHON" ]; then
  echo "ERRO: Python portátil não encontrado em $PYTHON"
  echo "Este projeto deve rodar a partir do HD externo MangabaAI."
  exit 1
fi

echo "==> Python: $($PYTHON --version) (portátil, no HD externo)"
echo "==> Iniciando API Gemma 4..."
echo "==> Swagger: http://localhost:8000/swagger"
echo ""

exec "$PYTHON" main.py
