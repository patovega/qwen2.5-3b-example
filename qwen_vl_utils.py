
import base64
from io import BytesIO
from PIL import Image

def process_vision_info(messages):
    image_inputs = []
    video_inputs = []
    for msg in messages:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content["type"] == "image":
                    # Si es una URL o ruta de archivo, la procesamos según corresponda
                    if isinstance(content["image"], str):
                        if content["image"].startswith("http://") or content["image"].startswith("https://"):
                            # Es una URL web, pero en realidad usaremos la imagen local
                            continue
                        elif content["image"].startswith("data:image/"):
                            # Es una imagen en base64
                            image_data = content["image"].split(",")[1]
                            image = Image.open(BytesIO(base64.b64decode(image_data)))
                            image_inputs.append(image)
                        else:
                            # Asumimos que es una ruta de archivo
                            try:
                                image = Image.open(content["image"])
                                image_inputs.append(image)
                            except:
                                print(f"No se pudo abrir la imagen: {content['image']}")
                    else:
                        # Asumimos que ya es un objeto PIL Image
                        image_inputs.append(content["image"])
                elif content["type"] == "video":
                    # Procesamiento de video (no implementado en este ejemplo)
                    pass
    return image_inputs, video_inputs
