import cv2
import numpy as np

def alinhar_e_extrair_embedding(image, app):
    # 1. Detecta as faces na imagem original usando a sua lógica
    faces = app.get(image)
    if len(faces) == 0:
        return None, None # Retorna None para o embedding e para a imagem

    face = faces[0]
    
    # 2. Pega as coordenadas dos olhos (kps[0] e kps[1])
    kps = face.kps
    olho_esquerdo = kps[0]
    olho_direito = kps[1]
    
    # 3. Calcula o ângulo de inclinação
    dy = olho_direito[1] - olho_esquerdo[1]
    dx = olho_direito[0] - olho_esquerdo[0]
    angulo = np.degrees(np.arctan2(dy, dx))
    
    # 4. Se a inclinação for muito pequena (ex: menor que 2 graus),
    # não precisa gastar processamento rotacionando a imagem.
    if abs(angulo) < 2.0:
        return face.embedding, image
        
    # 5. Caso contrário, rotaciona a imagem para alinhar os olhos
    centro_olhos = (
        int((olho_esquerdo[0] + olho_direito[0]) / 2),
        int((olho_esquerdo[1] + olho_direito[1]) / 2)
    )
    matriz_rotacao = cv2.getRotationMatrix2D(centro_olhos, angulo, scale=1.0)
    h, w = image.shape[:2]
    imagem_alinhada = cv2.warpAffine(image, matriz_rotacao, (w, h), flags=cv2.INTER_CUBIC)
    
    # 6. Extrai o embedding final a partir da imagem já alinhada e corrigida
    novas_faces = app.get(imagem_alinhada)
    if len(novas_faces) == 0:
        # Backup de segurança: se por algum motivo raro a rotação perder o rosto,
        # usamos o embedding da detecção original.
        return face.embedding, image
        
    return novas_faces[0].embedding, imagem_alinhada