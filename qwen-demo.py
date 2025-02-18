import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import gc
import requests
from PIL import Image
from io import BytesIO
import os
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
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        )
        print(f"Modelo cargado en: {device}")
        print(f"Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Memoria GPU usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        return model, processor
    except Exception as e:
        print(f"Error al inicializar el modelo: {str(e)}")
        raise

def process_image_query(model, processor, image_path, query_text, max_tokens=50):
    try:
        # Crear directorio para offload si no existe
        os.makedirs("offload_folder", exist_ok=True)
        
        # Cargar y procesar la imagen
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
            inputs = inputs.to("cuda")
 
            device = next(model.parameters()).device
            model_inputs = {k: v.to(device) for k, v in inputs.items()}
            response = ""
            # Mostrar uso de memoria
            if torch.cuda.is_available():
                print(f"Memoria GPU antes de inferencia: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            #manejo de memoria optimizado
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                  
                del output_text

            del model_inputs, inputs, image_inputs

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"Memoria GPU después de inferencia: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
            return response
        
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

def main():
    try:
        os.makedirs("offload_folder", exist_ok=True)
        
        # Configuración para usar GPU con precisión FP16
        model, processor = initialize_model(use_cpu=False)
        

        IMAGE_PATH = r"C:\Users\patri\Documents\ia\llm\qwen\images\anglo_58.jpg"
        QUERY = "provide only the identification number (ID) of the mining truck [unit/fleet number]. I only need the numeric or alphanumeric ID, required format: T-1234 or T-789 "
        
        response = process_image_query(model, processor, IMAGE_PATH, QUERY)
        print("\nRespuesta del modelo:", response[0])
        
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
        except:
            pass

if __name__ == "__main__":
    main()