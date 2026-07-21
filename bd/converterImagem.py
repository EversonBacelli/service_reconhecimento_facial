import base64
import cv2
import numpy as np

def converterImagem(imagem_texto):
    if not imagem_texto or not isinstance(imagem_texto, str):
        print("Erro: Entrada inválida para converterImagem")
        return None

    # Limpa o cabeçalho do Base64 se ele existir
    if "," in imagem_texto:
        imagem_texto = imagem_texto.split(",", 1)[1]

    try:
        dados = base64.b64decode(imagem_texto)
        
        img = cv2.imdecode(
            np.frombuffer(dados, np.uint8),
            cv2.IMREAD_COLOR
        )

        # SEGUNDA PROTEÇÃO: Verifica se o OpenCV conseguiu gerar a matriz da imagem
        if img is None:
            print("Erro: OpenCV não conseguiu decodificar os bytes da imagem. O Base64 pode estar corrompido.")
            return None

        # Agora é seguro converter de BGR para RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

    except Exception as e:
        print(f"Erro crítico na conversão da imagem: {e}")
        return None