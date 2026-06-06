#!/usr/bin/env bash
# Configura o ambiente do gemma4-api a partir do HD externo
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Verificando Python 3.10+..."
PYTHON=$(python3.11 -c 'import sys; print(sys.executable)' 2>/dev/null \
  || ~/.pyenv/versions/3.11.9/bin/python3.11 -c 'import sys; print(sys.executable)' 2>/dev/null \
  || python3 -c 'import sys; assert sys.version_info >= (3,10); print(sys.executable)' 2>/dev/null)

if [ -z "$PYTHON" ]; then
  echo "ERRO: Python 3.10+ não encontrado. Instale com:"
  echo "  curl https://pyenv.run | bash && pyenv install 3.11.9"
  exit 1
fi

echo "==> Usando: $PYTHON ($($PYTHON --version))"

echo "==> Criando venv..."
$PYTHON -m venv venv

echo "==> Instalando dependências..."
source venv/bin/activate
pip install --upgrade pip -q
pip install pydantic-settings -q
pip install -r requirements.txt -q

echo ""
echo "✓ Setup concluído!"
echo ""
echo "Para iniciar a API:"
echo "  cd $SCRIPT_DIR"
echo "  source venv/bin/activate"
echo "  python3 main.py"
echo ""
echo "Swagger: http://localhost:8000/swagger"
