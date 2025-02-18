from huggingface_hub import snapshot_download

# Nombre del modelo en Hugging Face
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

# Descargar todos los archivos del modelo
local_dir = snapshot_download(repo_id=model_name)

print(f"Modelo descargado en: {local_dir}")


