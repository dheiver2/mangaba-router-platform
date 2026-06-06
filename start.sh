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

# STAGE_TO_SSD=1 copia o modelo ativo para o SSD do notebook na 1ª carga
# (elimina o gargalo de leitura do USB). Padrão: ativado.
export STAGE_TO_SSD="${STAGE_TO_SSD:-1}"

echo "==> Python: $($PYTHON --version) (portátil, no HD externo)"
echo "==> Cache SSD: STAGE_TO_SSD=$STAGE_TO_SSD"
echo "==> Iniciando 🥭 Mangaba Router..."
echo "==> Swagger: http://localhost:8000/swagger"
echo ""

exec "$PYTHON" main.py
