# 🤖 Backend de Reconhecimento Facial

📌 **Visão Geral**

API REST desenvolvida em Python com **Flask** voltada para o **Processamento e Reconhecimento Facial**. O sistema utiliza técnicas avançadas de Visão Computacional e Aprendizado de Máquina (baseado em **InsightFace**, **OpenCV** e **NumPy**) para autenticação biométrica, análise de qualidade de imagem, correção de iluminação, alinhamento facial e busca por similaridade. Este trabalho foi realizado no contexto na disciplina de Processamento de Imagem e Visão Computacional no Curso de Mestrado na UFABC conduzido pelo professor Francisco de Assis Zampirolli.

📦 (Biblioteca de Reconhecimento Facial)

O InsightFace realiza o reconhecimento facial utilizando técnicas de Inteligência Artificial e Deep Learning. O processo pode ser dividido em três etapas principais: detecção e alinhamento do rosto, geração do embedding facial e comparação dos embeddings.

1 - Encontra e alinha os rostos
O primeiro passo é localizar o rosto dentro da imagem. Para isso, o InsightFace utiliza modelos de detecção facial baseados em Deep Learning, como o RetinaFace ou o SCRFD, dependendo do modelo e da configuração utilizada.
Esses modelos não procuram simplesmente por padrões fixos de pixels. Eles foram treinados com grandes quantidades de imagens para aprender quais características visuais indicam a presença de um rosto. Ao encontrar uma face, o detector também pode identificar pontos de referência faciais (landmarks). Um conjunto comum utiliza cinco pontos:
- olho esquerdo;
- olho direito;
- ponta do nariz;
- canto esquerdo da boca;
- canto direito da boca.

Esses pontos funcionam como referências para determinar a posição e a orientação do rosto. Por exemplo, se uma pessoa estiver com a cabeça inclinada, os olhos e a boca estarão inclinados na imagem original. Utilizando os landmarks, o sistema consegue realizar uma transformação geométrica para alinhar o rosto.
O objetivo é fazer com que rostos encontrados em imagens diferentes sejam colocados em uma posição semelhante antes de serem enviados para a rede neural. Normalmente, o rosto alinhado é convertido para um tamanho padronizado, como 112 × 112 pixels, dependendo do modelo utilizado.
<div align="center">
   <img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/03c093fb-7d8e-4f94-afde-1831a34fffd3" />
</div>
Essa etapa é importante porque reduz as diferenças causadas pela posição, inclinação e escala do rosto. Assim, duas fotografias da mesma pessoa podem apresentar condições diferentes — por exemplo, uma fotografia com o rosto ligeiramente inclinado e outra com o rosto reto — mas o sistema tenta entregar para a próxima etapa uma representação geometricamente mais consistente.

<br>

2 - Depois que o rosto foi detectado e alinhado, começa uma das etapas mais importantes do reconhecimento facial: a extração do embedding. O InsightFace utiliza modelos de redes neurais treinados especificamente para reconhecimento facial. Uma das abordagens associadas a esses modelos é o ArcFace, que utiliza redes neurais profundas para produzir uma representação numérica da face.
A imagem do rosto, normalmente alinhada em 112 × 112 pixels, é fornecida à rede neural. Então, ela analisa diferentes padrões presentes na face e  aprende, durante o treinamento, quais características são úteis para diferenciar uma pessoa da outra.
É importante entender que a rede não simplesmente transforma cada pixel da imagem em um número. Ela realiza várias operações internas através das camadas da rede neural e, ao final do processamento, produz uma representação compacta chamada embedding facial.
Em muitos modelos utilizados pelo InsightFace, esse embedding possui 512 valores:   Embedding = [e₁, e₂, e₃, ..., e₅₁₂]. Valores estes que representam, em conjunto, as características aprendidas pela rede neural sobre aquela face. Uma maneira melhor de visualizar isso é imaginar que cada pessoa, depois de passar pela rede neural, é representada por um ponto em um espaço de 512 dimensões.
Esse espaço é abstrato. Não conseguimos visualizar diretamente suas 512 dimensões, mas matematicamente podemos calcular a posição e a distância entre os embeddings.
<div align="center">
   <img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/27ecbc39-687f-4398-841d-804a345fcdc1" />
</div>

3 - Comparar Rostos
Depois que duas fotos de rostos são transformadas naquelas "impressões digitais numéricas" (listas de 512 números, chamadas de embeddings), o sistema precisa descobrir o quanto elas se parecem. Para isso, ele compara os 512 números de uma foto com os 512 números da outra, usando uma conta matemática chamada similaridade de cosseno. Pense nos embeddings como setas apontando para uma direção: se as duas setas apontam para direções parecidas, o resultado é uma similaridade alta, o que sugere que é a mesma pessoa; se apontam para direções bem diferentes, a similaridade é baixa, sugerindo pessoas diferentes. Existe também outra forma de comparar, calculando a "distância" entre os números — só que aqui a lógica é invertida: distância pequena indica que são parecidos (provável mesma pessoa), e distância grande indica que são diferentes.
Só o valor da similaridade ou da distância, porém, não é suficiente para decidir sozinho se é a mesma pessoa. É preciso definir um valor de referência, chamado threshold (limiar): se a similaridade for igual ou maior que esse valor, o sistema considera que é a mesma pessoa; se for menor, considera que são pessoas diferentes. Esse valor não é fixo — ele muda de acordo com o modelo usado e com o tipo de aplicação. Por exemplo, um sistema de segurança para liberar acesso a um prédio pode exigir um critério mais rígido do que um aplicativo que só organiza fotos no celular.
De forma resumida, todo o processo segue uma sequência de etapas: a imagem passa por detecção do rosto, depois é alinhada, em seguida é transformada em um embedding de 512 números pela rede neural ArcFace, e por fim dois desses embeddings são comparados matematicamente para gerar uma decisão final. Ou seja, o sistema não compara as fotos "olhando" pixel por pixel — ele compara representações numéricas dos rostos, e sua eficiência depende da capacidade da rede neural de gerar números parecidos para fotos da mesma pessoa e números diferentes para fotos de pessoas distintas.
<div align="center">
  <img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/218fd82b-20ae-4325-89f3-e1f2d90e2325" />
</div>

---

## 📋 Arquivo Principal (`main.py`)

O arquivo `main.py` atua como o *Controller* da aplicação, possuindo as seguintes responsabilidades:

1. **Inicialização e Configuração do Servidor**: Instancia a aplicação Flask;
2. **Carregamento dos Modelos de Reconhecimento**: Executa a carga inicial das redes neurais e algoritmos de extração facial na memória durante o startup do servidor, garantindo respostas de baixa latência nas requisições;
3. **Gerenciamento de Middlewares**: Habilita o CORS (*Cross-Origin Resource Sharing*) para integração com frontends web/mobile e gerencia exceções não capturadas globalmente com retornos padronizados em JSON.
4. **Registro de Rotas da API**:

### 🛣️ Endpoints da API

#### 1. Validar Qualidade da Imagem
- **Rota:** `/qualityImage`
- **Método:** `POST`
- **Descrição:** Valida se a imagem atende aos requisitos mínimos de qualidade (iluminação, foco, detecção de obstruções e presença de exatamente um rosto) utilizando o modelo InsightFace.
- **Resposta (JSON):** Status de aprovação (`sucesso`), mensagem descritiva e relatório detalhado das métricas analisadas.

#### 2. Comparar e Identificar Aluno
- **Rota:** `/comparar`
- **Método:** `POST`
- **Descrição:** Extrai o vetor de *embeddings* da imagem enviada via Base64 e calcula a **similaridade de cosseno** comparando com a lista de alunos cadastrados (`Aluno.ALUNOS`).
- **Resposta (JSON):** Dados do aluno correspondente e pontuação de *match* (caso a similaridade seja $\ge 0.70$); caso contrário, retorna mensagem de pessoa não encontrada.

#### 3. Corrigir Iluminação
- **Rota:** `/atualizarBrilho`
- **Método:** `POST`
- **Descrição:** Processa a imagem fornecida aplicando algoritmos adaptativos de equalização de histograma (CLAHE no espaço de cores LAB) para corrigir problemas de iluminação.
- **Resposta (JSON):** Imagem corrigida em formato Base64.

#### 4. Corrigir Rotação e Alinhamento
- **Rota:** `/corrigirRotacao`
- **Método:** `POST`
- **Descrição:** Detecta os *keypoints* anatômicos da face (olhos), calcula o ângulo de inclinação e aplica uma transformação afim para alinhar o rosto na horizontal, gerando o *embedding* otimizado.
- **Resposta (JSON):** Imagem alinhada codificada em Base64 e confirmação de processamento.

---

## 📦 Bibliotecas Externas

| Biblioteca | Finalidade no Projeto |
| :--- | :--- |
| **Flask** (`request`, `jsonify`) | Micro-framework web que sustenta a API, gerencia requisições HTTP e serializa respostas JSON. |
| **flask_cors** | Gerencia permissões do protocolo CORS, permitindo que clientes (React, Angular, Mobile) consumam a API com segurança. |
| **OpenCV** (`cv2`) | Processamento e manipulação de matrizes de imagens (espaços de cores, ajustes de iluminação, transformações geométrica). |
| **NumPy** (`np`) | Manipulação de vetores/matrizes multidimensionais de alta performance e cálculo matemático da **Similaridade de Cosseno** (`np.dot` e `np.linalg.norm`). |
| **json** | Módulo nativo para serialização, manipulação e estruturação de objetos JSON. |

---

## ⚙️ Funções Internas do Sistema

### 🔄 Conversão e Utilitários

* **`converter_para_base64(img_opencv)`**
  * **Entrada:** Imagem no formato `numpy.ndarray` (OpenCV).
  * **Retorno:** String no formato Base64 (`data:image/jpeg;base64,...`).
  * **Descrição:** Codifica a matriz de pixels em JPEG na memória e converte os bytes para Base64 com cabeçalho Data URL pronto para renderização no frontend.

* **`converterImagemColorida(imgBase64)`**
  * **Entrada:** String Base64.
  * **Retorno:** Imagem BGR (`numpy.ndarray`).
  * **Descrição:** Remove o cabeçalho Data URL, decodifica a string Base64 em dados binários brutos e os transforma em matriz tridimensional BGR via `cv2.imdecode`.

* **`buscarImagem(nome_imagem)`**
  * **Entrada:** Nome do arquivo de imagem salvo na base local.
  * **Retorno:** Conteúdo da imagem codificado em Base64.
  * **Descrição:** Constrói o caminho absoluto na pasta `bd/img_base64/`, realiza a leitura do arquivo texto e trata eventuais exceções `FileNotFoundError`.

### 🔍 Processamento e Visão Computacional

* **`validarImagem(img_cinza, img_colorida, app)`**
  * **Entradas:** Imagem em escala de cinza, imagem RGB/BGR e instância do modelo biométrico.
  * **Retorno:** Tupla `(aprovado: bool, relatorio: dict, embedding: np.ndarray)`.
  * **Descrição:** Executa testes sequenciais de iluminação (brilho médio), foco (variância do Laplaciano), obstrução de câmera e presença de exatamente uma face detectada.

* **`corrigir_iluminacao(img_colorida)`**
  * **Entrada:** Imagem colorida no formato BGR.
  * **Retorno:** Imagem BGR com iluminação equalizada.
  * **Descrição:** Converte o espaço de cores de BGR para **LAB**, isola o canal de luminosidade (*L*) e aplica a técnica **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*), preservando as tonalidades originais da foto.

* **`alinhar_e_extrair_embedding(img, app)`**
  * **Entradas:** Imagem colorida e instância do modelo biométrico.
  * **Retorno:** Tupla `(embedding, img_alinhada)`.
  * **Descrição:** Localiza os *keypoints* dos olhos. Se a inclinação da face for $> 2^\circ$, executa rotação afim (`cv2.warpAffine`) para alinhar a linha dos olhos na horizontal antes da extração biométrica. Possui mecanismo de *fallback*.

### 🧬 Biometria e Aprendizado Profundo

* **`extract_embedding(img)`**
  * **Entrada:** Imagem com face visível.
  * **Retorno:** Vetor numérico unidimensional (*embedding*).
  * **Descrição:** Submete a imagem à rede neural (**InsightFace/ArcFace**) para extrair a "assinatura biométrica" única de alta dimensão associada ao rosto.

* **`cosine_similarity(v1, v2)`**
  * **Entradas:** Vetor de características 1 (`v1`) e Vetor de características 2 (`v2`).
  * **Retorno:** Valor numérico de similaridade ($0.0$ a $1.0$).
  * **Descrição:** Calcula o cosseno do ângulo entre dois vetores através do produto escalar dividido pelo produto de suas normas euclidianas:
    $$\text{Similarity} = \frac{\mathbf{v_1} \cdot \mathbf{v_2}}{\|\mathbf{v_1}\| \|\mathbf{v_2}\|}$$

---

## 🏛️ Elementos Importantes da Arquitetura

### Classe `Aluno` (`bd/Aluno.py`)
Atua como o Modelo de Dados (*Model*) do domínio. 
- Gerencia a lista global em memória `Aluno.ALUNOS`.
- Inicializa estaticamente a rede neural `InsightFace` (`FaceAnalysis`).
- O método `@classmethod buscarDados()` carrega o arquivo `dados.json`, instancia os objetos da classe `Aluno` e os armazena para rápido acesso em memória durante o ciclo de vida da API.

### Base de Dados (`dados.json`)
Contém os registros estruturados dos estudantes integrando:
- **Metadados Cadastrais:** `matricula`, `nome`, `curso`, `turno`, `modulo` e `imagem`.
- **Vetor de Embeddings (`identificadores`):** Matriz densa de números em ponto flutuante gerada por aprendizado profundo que codifica os traços faciais do aluno.

---

## 📄 Licença
Este projeto está sob a licença **MIT** — veja o arquivo de licença para mais detalhes.

---

## 👤 Autor **Everson Bacelli** 
📧 **Contato:** everson.wilian29@gmail.com  
- 🐙 **GitHub:** [@EversonBacelli](https://github.com/EversonBacelli)
