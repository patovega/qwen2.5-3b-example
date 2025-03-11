import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gc
import requests
from PIL import Image
from io import BytesIO
import os
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qwen_vl_utils import process_vision_info

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

def process_image_query(model, processor, image_path, query_text, max_tokens=128):
    try:
        # Crear directorio para offload si no existe
        os.makedirs("offload_folder", exist_ok=True)
        image = download_image(image_path)
        
        if "qwen" in model.config.name_or_path.lower():
            # formato para Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
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
            
            # Manejo de memoria optimizado
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response = output_text
            
            # Limpiar memoria
            del generated_ids, generated_ids_trimmed
            del inputs, image_inputs

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return response
        else:
            raise ValueError(f"Modelo no soportado: {model.config.name_or_path}")
        
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

def extract_id_from_filename(filename):
    """Extrae el ID del camión del nombre del archivo."""
    # Patrón para capturar varios formatos:
    # - anglo_58.jpg (formato prefijo_número)
    # - cmdic_94.png (formato prefijo_número)
    # - centinela_17.jpg (formato prefijo_número)
    # - chiqui_g130.jpg (formato con letra después del prefijo)
    # - unknown_31.jpg (formato desconocido_número)
    # - rt_485.jpg (formato corto_número)
    
    # Primer intento: buscar prefijo_número
    pattern = r'([a-zA-Z]+)_(\d+|\w\d+)'
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        prefix = match.group(1).upper()
        id_number = match.group(2)
        
        # Normalizar el formato según el prefijo
        if prefix in ['ANGLO', 'CENTINELA', 'CMDIC', 'CMP', 'DET', 'DGM', 'DMH', 
                     'ESCONDIDA', 'MEL', 'PELAMBRES', 'RT', 'SG', 'UNKNOWN', 'CHUQUI']:
            return f"{prefix}-{id_number}"
        else:
            # Formato genérico si el prefijo no es reconocido
            return f"{prefix}-{id_number}"
    
    # Si no encuentra ese patrón, intentar otros formatos
    # Formato T-123 o CAM-456 que pueda estar en el nombre
    pattern2 = r'(T|CAM)-\d+'
    match2 = re.search(pattern2, filename, re.IGNORECASE)
    if match2:
        return match2.group(0).upper()
    
    # Último intento: cualquier letra seguida de números al final
    pattern3 = r'([a-zA-Z]+)(\d+)\.'
    match3 = re.search(pattern3, filename, re.IGNORECASE)
    if match3:
        return f"{match3.group(1).upper()}-{match3.group(2)}"
    
    return None

def extract_numbers(input_string):
    """
    Extrae solo los números de un string y elimina ceros iniciales
    
    Args:
        input_string (str): String de entrada
    
    Returns:
        str: String con solo números y sin ceros iniciales
    """
    only_numbers = re.sub(r'[^0-9]', '', input_string)
    
    if not only_numbers:
        return ''
    
    without_leading_zeros = str(int(only_numbers))
    
    return without_leading_zeros
 

def process_image_batch(folder_path, model, processor, query_text, max_images=65):
    """Procesa un lote de imágenes y compara los IDs inferidos con los nombres de archivo."""
    results = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f)) and 
                  os.path.splitext(f)[1].lower() in supported_extensions]
    image_files = image_files[:max_images]
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)
        print(f"\nProcesando imagen {idx+1}/{len(image_files)}: {img_file}")
        expected_id = extract_id_from_filename(img_file)
        expected_id = extract_numbers(expected_id)

        try:
            response = process_image_query(model, processor, img_path, query_text)
            predicted_id = None
            if response and len(response) > 0:
                predicted_id = extract_numbers(response[0])
                print(f"  ID inferido: {predicted_id}")
                print(f"  ID esperado: {expected_id}")

            match = predicted_id == expected_id if predicted_id and expected_id else False
            results.append({
                'filename': img_file,
                'image_path': img_path,
                'expected_id': expected_id,
                'predicted_id': predicted_id,
                'match': match
            })
            
        except Exception as e:
            print(f"Error procesando {img_file}: {e}")
            results.append({
                'filename': img_file,
                'image_path': img_path,
                'expected_id': expected_id,
                'predicted_id': None,
                'match': False,
                'error': str(e)
            })
    
    return results

def visualize_results(results):
    """Crea una visualización de los resultados en un grid."""
    n_images = len(results)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(16, 4 * rows))
    gs = gridspec.GridSpec(rows, cols)
    
    for i, result in enumerate(results):
        ax = plt.subplot(gs[i])
        
        try:
            img = Image.open(result['image_path'])
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, "Error loading image", 
                    ha='center', va='center', transform=ax.transAxes)

        title = f"Esperado: {result['expected_id'] or 'N/A'}\n"
        title += f"Predicho: {result['predicted_id'] or 'N/A'}\n"
        
        if result['match']:
            title += "✓ COINCIDE"
            title_color = 'green'
        else:
            title += "✗ NO COINCIDE"
            title_color = 'red'
        
        ax.set_title(title, color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    
    results_dir = "resultados"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "resultados_inferencia.png"), dpi=150)
    plt.savefig(os.path.join(results_dir, "resultados_inferencia.pdf"))
    
    print(f"\nResumen guardado en carpeta '{results_dir}'")
    print(f"Total procesado: {len(results)}")
    print(f"Coincidencias: {sum(1 for r in results if r['match'])}")
    print(f"Errores: {sum(1 for r in results if r['match']==False)}")
    print(f"Efectividad: {(sum(1 for r in results if r['match']) / len(results) * 100):.2f}%")
    return plt

def main():
    try:
        os.makedirs("offload_folder", exist_ok=True)
        model, processor = initialize_model(use_cpu=False)
        
        # Carpeta con imágenes
        IMAGES_FOLDER = r"C:\Users\patri\Documents\ia\llm\qwen\images"
        
        # PROMT
        QUERY = "Provide only the identification number (ID) of the mining truck [unit/fleet number]. I only need the numeric or alphanumeric ID. Exclude 797, 797B, 930E, 930, 965,650, 791 as these are truck model numbers, not identification numbers."
        results = process_image_batch(IMAGES_FOLDER, model, processor, QUERY)

        visualize_results(results)
        report_path = os.path.join("resultados", "informe_detallado.txt")
        with open(report_path, 'w') as f:
            f.write("INFORME DE INFERENCIA DE IDs DE CAMIONES MINEROS\n")
            f.write("="*50 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Imagen {i+1}: {result['filename']}\n")
                f.write(f"  ID esperado: {result['expected_id'] or 'No detectado'}\n")
                f.write(f"  ID predicho: {result['predicted_id'] or 'No detectado'}\n")
                f.write(f"  Coincidencia: {'SÍ' if result['match'] else 'NO'}\n")
                if result.get('error'):
                    f.write(f"  Error: {result['error']}\n")
                f.write("\n")
                
            # Estadísticas
            matches = sum(1 for r in results if r['match'])
            total = len(results)
            accuracy = (matches / total) * 100 if total > 0 else 0
            
            f.write("\nESTADÍSTICAS\n")
            f.write("="*20 + "\n")
            f.write(f"Total imágenes: {total}\n")
            f.write(f"Coincidencias: {matches}\n")
            f.write(f"Precisión: {accuracy:.2f}%\n")
            
        print(f"Informe detallado guardado en: {report_path}")
        
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