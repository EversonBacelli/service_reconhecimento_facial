import json
import os

def buscarImagem(imagem):
    # Abre o arquivo JSON para leitura
    # Descobre o caminho absoluto da pasta onde o getDados.py está
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    pasta_atual = os.getcwd()
    
    doc = os.path.join(pasta_atual, 'bd/img_base64', imagem)
  
    try:
        with open(doc, 'r', encoding='utf-8') as arquivo:
            conteudo = arquivo.read()  # <--- Aqui o texto é salvo na variável 'conteudo'
    
        # Agora você pode usar a variável normalmente
        return conteudo

    except FileNotFoundError:
        print(f"Erro: O arquivo no caminho {doc} não foi encontrado.")

    # for aluno in estudantes:
    #     print(f"Nome: {aluno['nome']}")
    #     print(f"Curso: {aluno['curso']} ({aluno['turno']})")
    #     print(f"Identificador EMB1: {aluno['identificadores']['emb1']}")
    #     print("-" * 30)