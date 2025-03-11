import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gc
import requests
from PIL import Image
from io import BytesIO
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def download_image(image_path):
    try:
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            
        return image
    except Exception as e:
        print(f"Error al cargar la imagen: {str(e)}")
        raise

def initialize_model(model_name="Qwen/Qwen2.5-VL-3B-Instruct", use_cpu=False):
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
        processor = AutoProcessor.from_pretrained(model_name)
 
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        print(f"Modelo cargado en: {device}")
        if torch.cuda.is_available():
            print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Memoria GPU usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        return model, processor
    except Exception as e:
        print(f"Error al inicializar el modelo: {str(e)}")
        raise

def process_vision_info(messages):

    images = []
    videos = []
    
    for message in messages:
        if "content" in message:
            for content in message["content"]:
                if content.get("type") == "image":
                    images.append(content["image"])
                elif content.get("type") == "video":
                    videos.append(content["video"])
    
    return images, videos

def classify_toy_truck(model, processor, image_path1, image_path2, max_tokens=128):
    try:
        os.makedirs("offload_folder", exist_ok=True)
        
        # Cargar ambas imágenes
        image1 = download_image(image_path1)
        image2 = download_image(image_path2)
        
        # Prompt para clasificación
        query_text = "Analiza estas dos imágenes de camiones mineros. Identifica cuál es un camión de juguete y cuál es un camión real. Si alguno de los camiones tiene un número de identificación visible, indícalo. Proporciona tu respuesta en formato: 'Imagen 1: [juguete/real], Número: [si es visible], Imagen 2: [juguete/real], Número: [si es visible]'. Explica brevemente por qué puedes diferenciarlos."
        
        # Formato para Qwen2.5-VL con múltiples imágenes
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image1,
                    },
                    {
                        "type": "image",
                        "image": image2,
                    },
                    {"type": "text", "text": query_text},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        device = next(model.parameters()).device
        
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text
        
        del generated_ids, generated_ids_trimmed
        del inputs, image_inputs

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return response[0] if response else "", image1, image2
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Error: CUDA sin memoria. Intentando con batch más pequeño o en CPU.")
            raise
        else:
            print(f"Error en tiempo de ejecución: {str(e)}")
            raise
    except Exception as e:
        print(f"Error al procesar la consulta: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def visualize_classification(image1, image2, result_text):
    """Visualiza las imágenes con el resultado de la clasificación."""
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title("Imagen 1", fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title("Imagen 2", fontsize=14)
    plt.axis('off')
    

    import textwrap
    wrapped_text = textwrap.fill(result_text, width=80)
    
    plt.figtext(0.5, 0.05, wrapped_text, 
                ha="center", va="center", fontsize=12, 
                bbox={"facecolor":"white", "alpha":0.9, "pad":10, "edgecolor":"lightgray"})
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    results_dir = "resultados"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "clasificacion_camiones.png"), dpi=150, bbox_inches='tight')
    
    print(f"\nResultados guardados en carpeta '{results_dir}'")
    return plt

def main():
    try:
        os.makedirs("offload_folder", exist_ok=True)
        model, processor = initialize_model(use_cpu=False)
        
        # Rutas de las imágenes (puedes cambiar estas rutas)
        imagen1_path = "images\\truck_toys_example\\t1.png"
        imagen2_path = "images\\dgm_13.jpg"
        
        print("\nClasificando imágenes...")
        resultado, img1, img2 = classify_toy_truck(model, processor, imagen1_path, imagen2_path, max_tokens=256)
        
        print("\nResultado de la clasificación:")
        print(resultado)
        
        visualize_classification(img1, img2, resultado)
        
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import shutil
            if os.path.exists("offload_folder"):
                shutil.rmtree("offload_folder")
        except Exception as cleanup_error:
            print(f"Error al limpiar archivos temporales: {cleanup_error}")

if __name__ == "__main__":
    main()