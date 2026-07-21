import cv2

def corrigir_iluminacao(img_colorida):
    # Converte para o espaço de cores LAB para separar a luminosidade (L) dos canais de cor (A e B)
    lab = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplica o CLAHE no canal de luminosidade
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Junta os canais novamente e converte de volta para BGR
    imagem_corrigida = cv2.merge((cl, a, b))
    return cv2.cvtColor(imagem_corrigida, cv2.COLOR_LAB2BGR)