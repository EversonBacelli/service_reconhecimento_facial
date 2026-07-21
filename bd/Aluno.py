
import os
import json
from bd.getDados import buscarImagem
from bd.converterImagem import converterImagem
from bd.extract import extract_embedding
from insightface.app import FaceAnalysis

class Aluno:
    ALUNOS = []
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CPUExecutionProvider']
    )

    app.prepare(ctx_id=0)
    providers=['CUDAExecutionProvider']




    def __init__(self, matricula, curso, turno, modulo, nome, identificador,arquivo):
        self.matricula = matricula
        self.curso = curso
        self.turno = turno
        self.modulo = modulo
        self.nome = nome
        self.identificador = identificador
        # Estrutura de identificadores conforme o JSON
        # imagem = buscarImagem(arquivo)
        # self.imagem = converterImagem(imagem)

        # # 2. PROTEÇÃO: Só extrai o embedding se a imagem foi convertida com sucesso
        # if self.imagem is not None:
        #     self.embed = extract_embedding(self.imagem, Aluno.app)
        # else:
        #     # Se a imagem for None, define o embed como None e avisa no terminal
        #     print(f"⚠️ [Aviso] Falha ao converter imagem do aluno {self.nome}. O registro foi pulado.")
        #     self.embed = None

        #self.embed = extract_embedding(self.imagem, Aluno.app)
        Aluno.ALUNOS.append(self)

    def __repr__(self):
        """Define como o objeto será visualizado ao dar um print()"""
        return f"<Aluno: {self.nome} - Matrícula: {self.matricula}>"

    @classmethod
    def buscarDados(cls):
        """
        Método de classe que lê o arquivo 'dados.json' de forma dinâmica
        e retorna uma lista de objetos do tipo 'Aluno'.
        """
        # Descobre o caminho da pasta onde este arquivo (aluno.py) está
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        
        # Define o caminho do arquivo JSON (ajuste o caminho 'dados.json' ou '../dados.json' se necessário)
        caminho_json = os.path.join(diretorio_atual, 'dados.json')

        try:
            with open(caminho_json, 'r', encoding='utf-8') as arquivo:
                dados_json = json.load(arquivo)
            
            for item in dados_json:
                # Instancia um novo objeto Aluno mapeando os dados do JSON
                # print(item)
                Aluno(
                    item['matricula'],
                    item['curso'],
                    item['turno'],
                    item['modulo'],
                    item['nome'],
                    item['identificadores'],
                    item['imagem']
                )
                
                
                
        except FileNotFoundError:
            print(f"Erro: O arquivo {caminho_json} não foi encontrado.")
            return []