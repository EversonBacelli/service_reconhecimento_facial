# Libs 
from flask import Flask, request, jsonify
import json
import cv2
import numpy as np
from flask_cors import CORS


# funções Internas
from similary import cosine_similarity
from bd.Aluno import Aluno
from valida import validarImagem
from bd.converterImagemColorida import converterImagemColorida
from bd.converterImagem import converterImagem
from bd.extract import extract_embedding
from correcaoBrilho import corrigir_iluminacao
from converterBase64 import converter_para_base64
from correcaoRotacao import alinhar_e_extrair_embedding

# Libs 
from flask import Flask, request, jsonify
import json
import cv2
import numpy as np
from flask_cors import CORS

# carregar dados
Aluno.buscarDados()
alunos = Aluno.ALUNOS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes


@app.route("/qualityImage", methods=["POST"])
def quality():
    try:
        data = request.get_json()
        base64_data = data["imagemBase64"]
        
        # 1. Gera as duas versões da imagem recebida do front-end
        img_colorida = converterImagemColorida(base64_data)
        img_cinza = converterImagem(base64_data)
        
        # 2. Chama a validação passando o seu objeto 'app' do InsightFace
        aprovado, detalhes, embedding = validarImagem(img_cinza, img_colorida, Aluno.app)

        if aprovado:
            # SUCESSO: A imagem passou em tudo e você já tem o embedding!
            # Aqui você pode salvar o embedding ou usá-lo para reconhecer a pessoa.
            return jsonify({
                "sucesso": True,
                "mensagem": "Imagem validada com sucesso!",
                "relatorio": detalhes
            }), 200
        else:
            # FALHA: Repassa o relatório com o motivo específico (brilho, foco, etc.)
            return jsonify({
                "sucesso": False,
                "mensagem": "A imagem não atende aos requisitos mínimos de qualidade.",
                "relatorio": detalhes
            }), 400

    except Exception as e:
        return jsonify({"sucesso": False, "mensagem": f"Erro interno: {str(e)}"}), 500


@app.route('/comparar', methods=['POST'])
def verify():
    data = request.get_json()
    data = data["imagemBase64"]
    
    imagem = converterImagem(data)
    emb1 = extract_embedding(imagem, Aluno.app)


    for aluno in Aluno.ALUNOS:
        emb2 = np.array(aluno.identificador)
        v1 = emb1.flatten()
        v2 = emb2.flatten()


        similarity = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        ) 

        if similarity >= 0.70:  

            return jsonify(
                {
                    'match': similarity
                }, 
                {
                    'aluno': aluno.__dict__
                }
            )
    
    return jsonify({
        'NoMatch': 'Pessoa Não Encontrada'
    })


@app.route('/atualizarBrilho', methods=['POST'])
def corrigir():
    try:
        data = request.get_json()
        base64_data = data["imagemBase64"]
        
        # 1. Gera as duas versões da imagem recebida do front-end
        img_colorida = converterImagemColorida(base64_data)
        # img_cinza = converterImagem(base64_data)
        
        nova_img = corrigir_iluminacao(img_colorida)
        nova_img = converter_para_base64(nova_img)
        
        return jsonify({
                "mensagem": "Bilho atualizado",
                "imagemBase64": nova_img
            }), 200
        

    except Exception as e:
        return jsonify({"sucesso": False, "mensagem": f"Erro interno: {str(e)}"}), 500


@app.route('/corrigirRotacao', methods=['POST'])
def corrigir_rotacao():
    try:
        data = request.get_json()
        base64_data = data["imagemBase64"]

        # 1. Converte Base64 -> OpenCV (ndarray)
        img_colorida = converterImagemColorida(base64_data)
        
        # 2. Executa o detector do InsightFace para achar o rosto e os KPS (olhos)
        embedding, img_alinhada = alinhar_e_extrair_embedding(img_colorida, Aluno.app)
        
        
        img_alinhada_base64 = converter_para_base64(img_alinhada)
        
        return jsonify({
            "sucesso": True,
            "imagemBase64": img_alinhada_base64
        }), 200
        
    except Exception as e:
        return jsonify({"sucesso": False, "mensagem": f"Erro interno: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)





