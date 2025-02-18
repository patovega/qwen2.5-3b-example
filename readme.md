# Mining Truck ID Recognition with Qwen 2.5
Este proyecto implementa un sistema de reconocimiento automático de números identificadores en camiones mineros utilizando el modelo multimodal Qwen 2.5.

## Toda la info de Qwen 2.5 en:
https://github.com/QwenLM/Qwen2.5

## Los modelos de Qwen 2.5 los puedes descargar desde Hugging Face
https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

Este código tiene como base el artículo expuesto en Hugging Face.

## Descarga del Modelo
El modelo Qwen 2.5 se puede descargar directamente desde Hugging Face usando el siguiente script: (downloader.py)

```python
from huggingface_hub import snapshot_download
# Nombre del modelo en Hugging Face
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
# Descargar todos los archivos del modelo
local_dir = snapshot_download(repo_id=model_name)
print(f"Modelo descargado en: {local_dir}")
```

Alternativamente, puedes descargarlo mediante línea de comandos:
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instructt --local-dir ./models/Qwen2.5-VL-3B-Instruct
```

💡 **Nota**: Asegúrate de tener suficiente espacio en disco, ya que el modelo ocupa varios GB. tambien GPU necesaria, la mia es de 6gb.
 
## Descripción
El sistema procesa imágenes de camiones mineros y extrae automáticamente el número identificador pintado en el costado del equipo. Utiliza Qwen 2.5, un modelo multimodal avanzado que combina procesamiento de texto e imágenes con capacidades visuales mejoradas del equipo de ALIBABA CLOUD

## Requisitos
- Python 3.8+
- PyTorch
- Transformers
- OpenCV
- ReportLab
- PIL
- CUDA compatible GPU (Probado en NVIDIA RTX 3060)

## Estructura del Proyecto
```
├── models/
│   └── Qwen2.5-7B-Instruct/  # Modelo Qwen (no incluido en el repo)
├── images/                    # Carpeta con imágenes de prueba
├── images_all/                    # Carpeta con mas imagenes de prueba
├── qwen-demo-folder.py                    # Script principal, procesa la carpeta images
├── qwen-demo.py                    # script para el ejemplo simple de de qwen
└── README.md
```

## Uso
1. Coloca las imágenes de los camiones en la carpeta `images/`
2. Ejecuta el script:
```bash
python qwen-demo-folder.py
```
3. El script generará un PDF con:
   - Visualización de resultados
   - Estadísticas de precisión
   - Lista de errores encontrados
 
## Características
- Procesamiento de imágenes de hasta 1024x1024 píxeles
- Soporte para múltiples formatos de imagen
- Generación de reportes en PDF
- Post-procesamiento de resultados
- Manejo de casos especiales (números prohibidos)
- Estadísticas de precisión

## Consideraciones
Para obtener mejores resultados:
- Las imágenes deben tener buena iluminación
- El número debe ser claramente visible
- Evitar ángulos extremos
- Mantener una distancia consistente

## Limitaciones Conocidas
- Sensible a condiciones de iluminación extremas
- Puede confundirse con números similares
- Requiere que el número sea visible y legible

## Ventajas de Qwen 2.5
- Mayor resolución de entrada (hasta 1024x1024)
- Mejor comprensión contextual visual
- Menor latencia que modelos competidores
- Soporte para múltiples idiomas
- Mayor precisión en condiciones adversas