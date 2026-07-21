import cv2
import numpy as np
import base64

def converter_para_base64(img_opencv):
    # Codifica a imagem de volta para formato JPEG na memória
    sucesso, buffer = cv2.imencode('.jpg', img_opencv)
    if not sucesso:
        raise ValueError("Não foi possível codificar a imagem do OpenCV.")
        
    # Converte os bytes do buffer para Base64 binário e depois para string UTF-8
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Retorna formatado para o HTML/Front-end ler direto no <img src="...">
    return f"data:image/jpeg;base64,{img_base64}"