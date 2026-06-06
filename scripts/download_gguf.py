"""Baixa os GGUF Q4_0 oficiais (QAT) do Google — quantizados, rodam em 16GB.
Cada modelo: arquivo principal .gguf + mmproj (projetor de visão multimodal).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()
TOK = os.getenv("HF_TOKEN") or None

# (repo, arquivo_principal, arquivo_mmproj_visao)
GGUF = {
    "e2b": ("google/gemma-4-E2B-it-qat-q4_0-gguf",   "gemma-4-E2B_q4_0-it.gguf",        "gemma-4-E2B-it-mmproj.gguf"),
    "e4b": ("google/gemma-4-E4B-it-qat-q4_0-gguf",   None,                              None),  # resolvido em runtime
    "12b": ("google/gemma-4-12B-it-qat-q4_0-gguf",   "gemma-4-12b-it-qat-q4_0.gguf",    "mmproj-gemma-4-12b-it-qat-q4_0.gguf"),
    # 26B removido: 15GB não cabe nos 16GB de RAM mesmo quantizado.
}

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mangaba_models")


def _resolve_e4b():
    from huggingface_hub import list_repo_files
    repo = GGUF["e4b"][0]
    files = [f for f in list_repo_files(repo, token=TOK) if f.endswith(".gguf")]
    main = next((f for f in files if "mmproj" not in f.lower()), None)
    proj = next((f for f in files if "mmproj" in f.lower()), None)
    return repo, main, proj


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = sys.argv[1:] or list(GGUF.keys())
    for slug in targets:
        repo, main_f, proj_f = GGUF[slug]
        if main_f is None:
            repo, main_f, proj_f = _resolve_e4b()
        dest = os.path.join(OUT, slug)
        os.makedirs(dest, exist_ok=True)
        for f, tag in [(main_f, "q4_0"), (proj_f, "mmproj")]:
            if not f:
                continue
            print(f"[{slug}] baixando {f} ...")
            path = hf_hub_download(repo_id=repo, filename=f, local_dir=dest, token=TOK)
            # renomeia para a marca Mangaba
            branded = os.path.join(dest, f"mangaba-{slug}-{tag}.gguf")
            if os.path.abspath(path) != os.path.abspath(branded):
                os.replace(path, branded)
        print(f"[{slug}] OK -> {dest} (marca Mangaba)")
    print("\nModelos Mangaba prontos.")


if __name__ == "__main__":
    main()
