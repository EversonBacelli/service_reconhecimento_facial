import cv2
import numpy as np

def validarImagem(img_cinza, img_colorida, app):
    """
    Realiza a validação completa e, se aprovada, extrai o embedding do rosto.
    
    :return: (imagem_valida, relatorio, embedding)
    """
    if img_cinza is None or img_colorida is None:
        return False, {
            "geral": "Erro ao carregar a imagem.",
            "brilho": "Não avaliado",
            "foco": "Não avaliado",
            "presenca_rosto": "Não avaliado",
            "conteudo": "Não avaliado"
        }, None

    relatorio = {
        "brilho": "Aprovado",
        "foco": "Aprovado",
        "presenca_rosto": "Aprovado",
        "conteudo": "Aprovado"
    }
    
    imagem_valida = True
    embedding_extraido = None

    # --- 1. TESTE DE BRILHO ---
    brilho_medio = np.mean(img_cinza)
    if brilho_medio < 40:
        relatorio["brilho"] = "Recusado: Foto muito escura."
        imagem_valida = False
    elif brilho_medio > 220:
        relatorio["brilho"] = "Recusado: Foto muito clara/estourada."
        imagem_valida = False

    # --- 2. TESTE DE FOCO ---
    variancia_foco = cv2.Laplacian(img_cinza, cv2.CV_64F).var()
    if variancia_foco < 100:
        relatorio["foco"] = "Recusado: Imagem borrada."
        imagem_valida = False

    # --- 3. TESTE DE CONTEÚDO (Morfologia) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradiente = cv2.morphologyEx(img_cinza, cv2.MORPH_GRADIENT, kernel)
    pixels_borda = np.sum(gradiente > 50) 
    total_pixels = img_cinza.shape[0] * img_cinza.shape[1]
    proporcao_bordas = (pixels_borda / total_pixels) * 100

    if proporcao_bordas < 1.5: 
        relatorio["conteudo"] = "Recusado: Câmera tampada ou imagem vazia."
        imagem_valida = False

    # --- 4. TESTE DE PRESENÇA DE ROSTO & EXTRAÇÃO DE EMBEDDING ---
    if imagem_valida:
        try:
            # Executa o app.get() uma única vez!
            faces = app.get(img_colorida)
            
            if len(faces) == 0:
                relatorio["presenca_rosto"] = "Recusado: Nenhum rosto identificado."
                imagem_valida = False
            elif len(faces) > 1:
                relatorio["presenca_rosto"] = "Recusado: Mais de um rosto detectado."
                imagem_valida = False
            else:
                # Se houver exatamente 1 rosto, salvamos o embedding dele
                embedding_extraido = faces[0].embedding
                
        except Exception as e:
            relatorio["presenca_rosto"] = f"Erro na IA: {str(e)}"
            imagem_valida = False
    else:
        relatorio["presenca_rosto"] = "Não avaliado devido a falhas prévias."

    return imagem_valida, relatorio, embedding_extraido