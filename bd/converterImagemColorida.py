import base64
import cv2
import numpy as np

def converterImagemColorida(base64_string):
    # Remove cabeçalhos comuns de dados Base64 se existirem (ex: "data:image/jpeg;base64,")
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decodifica os bytes do base64
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decodifica os bytes diretamente para uma imagem colorida (formato BGR padrão do OpenCV)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_color