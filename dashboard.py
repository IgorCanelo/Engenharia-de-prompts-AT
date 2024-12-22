import streamlit as st
import yaml
import json
import pandas as pd
import altair as alt
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Função para criar o modelo e o índice
def criar_indice(texts, model_name='neuralmind/bert-base-portuguese-cased', llm_model_dir='../data/bertimbau/'):
    # Carregar o modelo de embeddings
    embedding_model = SentenceTransformer(
        model_name, 
        cache_folder=llm_model_dir, 
        device='cpu'
    )
    
    # Converter as strings (proposições e dados dos deputados) para embeddings
    embeddings = embedding_model.encode(texts)
    
    # Converter os embeddings para um array NumPy, necessário para Faiss
    embeddings = np.array(embeddings).astype("float32")
    
    # Criar o índice Faiss (usando a distância Euclidiana)
    d = embeddings.shape[1]  # Dimensão dos embeddings
    index = faiss.IndexFlatL2(d)
    
    # Adicionar os embeddings ao índice
    index.add(embeddings)
    
    return embedding_model, index

# Função para responder perguntas usando o índice
def responder_pergunta(query, embedding_model, index, texts, k=2):
    # Converter a consulta do usuário para embeddings
    query_embedding = embedding_model.encode([query]).astype("float32")
    
    # Fazer a busca no índice
    distances, indices = index.search(query_embedding, k)
    
    # Retornar os resultados mais relevantes
    respostas = []
    for i in range(k):
        resposta = {
            "texto": texts[indices[0][i]],
            "distancia": distances[0][i]
        }
        respostas.append(resposta)
    
    return respostas

# Função principal para o chat no Streamlit
def chat_assistente_virtual():
    # Lista de textos (substitua com dados reais de deputados e proposições)

    # Carregar os DataFrames
    df_deputados = pd.read_parquet("data/deputados.parquet")
    df_despesas_deputados = pd.read_parquet("data/despesas_deputados.parquet")
    df_preposicoes_deputados = pd.read_parquet("data/proposicoes_deputados.parquet")

    valorliq = df_despesas_deputados.groupby("idDeputado")['valorLiquido'].sum()
    despesa = df_despesas_deputados.groupby("tipoDespesa")['idDeputado'].count()  # Alteração para contar os registros por tipo de despesa
    preposicoes = df_preposicoes_deputados.groupby("codTema").size()  # Alteração para contar
    # Processar e combinar os dados
    # Converter os valores de valorliq para uma lista de textos
    textos_valorliq = [f"Deputado {id_deputado} gastou R${valor:.2f} em despesas." for id_deputado, valor in valorliq.items()]

    # Converter as despesas para uma lista de textos
    textos_despesa = [f"Tipo de despesa: {tipo} teve {quantidade} registros." for tipo, quantidade in despesa.items()]


    # Converter as proposições para uma lista de textos
    textos_preposicoes = [f"Código do tema {codigo_tema} tem {quantidade} proposições." for codigo_tema, quantidade in preposicoes.items()]

    # Exemplo de texto com informações dos deputados
    textos_deputados = df_deputados.apply(lambda row: f"Deputado {row['id']} - {row['siglaPartido']} Sigla do partido", axis=1).tolist()

    # Combinar todos os textos
    texts = textos_valorliq + textos_despesa + textos_preposicoes + textos_deputados

    # Criar o modelo e o índice uma vez
    embedding_model, index = criar_indice(texts)
    
    # Interface de chat com Streamlit
    st.title("Chat com o Assistente Virtual da Câmara dos Deputados")
    
    # Criar o input de texto para o usuário
    usuario_input = st.text_input("Digite sua pergunta:")

    if usuario_input:
        # Obter as respostas mais relevantes
        respostas = responder_pergunta(usuario_input, embedding_model, index, texts)
        
        st.write("Resposta do Assistente Virtual:")
        for resposta in respostas:
            st.write(f"- {resposta['texto']} (distância: {resposta['distancia']:.4f})")


# Função para carregar o conteúdo do arquivo YAML
def carregar_yaml(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Arquivo {caminho_arquivo} não encontrado.")
        return None
    except yaml.YAMLError as e:
        st.error(f"Erro ao ler o arquivo YAML: {e}")
        return None

# Função para carregar o conteúdo do arquivo JSON
def carregar_json(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'r', encoding='ISO-8859-1') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error(f"Arquivo {caminho_arquivo} não encontrado.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Erro ao ler o arquivo JSON: {e}")
        return None

# Função para exibir o título e descrição da solução na página Overview
def exibir_titulo_descricao():
    st.title("Overview análise de deputados")
    st.write("Esta é uma solução de análise de distribuição de deputados, abordando tópicos como despesas e proposições.")

# Função para exibir o painel de resumo do YAML
def exibir_resumo_yaml():
    config = carregar_yaml("data/config.yaml")  # Coloque o caminho correto para o seu arquivo
    config = config.get("overview_summary")
    if config:
        st.subheader("Resumo do Configuração")
        st.text_area("Resumo do Arquivo YAML", config, height=200)

# Função para exibir o gráfico de barras
def exibir_grafico_barras():
    imagem = Image.open("docs/distribuicao_deputados.png")  # Caminho correto para a imagem
    st.image(imagem, caption="Distribuição de Deputados", use_container_width=True)

# Função para exibir os insights extraídos do arquivo JSON
def exibir_insights_json():
    insights = carregar_json("data/insights_distribuicao_deputados.json")  # Caminho correto para o arquivo JSON
    if insights:
        st.subheader("Insights da Análise")
        for key, value in insights.items():
            st.write(f"**{key}:** {value}")

# Função para exibir as informações de Despesas
def exibir_despesas():
    st.title("Despesas dos Deputados")
    
    # Carregar dados de despesas diárias
    df_despesas = pd.read_parquet("data/despesas_deputados.parquet")  # Caminho correto para o arquivo parquet
    deputados = df_despesas['idDeputado'].unique()
    
    # Selectbox para escolha do deputado
    deputado_selecionado = st.selectbox("Selecione o ID do Deputado", deputados)
    
    # Filtrar dados para o deputado selecionado
    df_despesas_selecionado = df_despesas[df_despesas['idDeputado'] == deputado_selecionado]
    
    # Exibir gráfico de barras com série temporal de despesas
    st.subheader(f"Despesas do Deputado {deputado_selecionado}")
    grafico = alt.Chart(df_despesas_selecionado).mark_bar().encode(
        x='dataDocumento',
        y='tipoDespesa',
        color='tipoDespesa'
    ).properties(title="Série Temporal de Despesas")
    st.altair_chart(grafico, use_container_width=True)
    
    # Exibir insights sobre as despesas
    insights_despesas = carregar_json("data/insights_despesas_deputados.json")
    if insights_despesas:
        st.subheader(f"Insights sobre as Despesas")
        st.write(insights_despesas.get("insights"))

# Função para exibir as proposições
def exibir_proposicoes():
    st.title("Proposições dos Deputados")
    
    # Carregar dados de proposições
    df_proposicoes = pd.read_parquet("data/proposicoes_deputados.parquet")  # Caminho correto para o arquivo parquet
    
    # Exibir tabela de proposições
    st.subheader("Tabela de Proposições")
    st.dataframe(df_proposicoes)
    
    # Carregar resumo das proposições
    resumo_proposicoes = carregar_json("data/sumarizacao_proposicoes.json")
    if resumo_proposicoes:
        st.subheader("Resumo das Proposições")
        st.write(resumo_proposicoes)

    # Adicionar o chat do assistente virtual
    chat_assistente_virtual()

# Função da página "Overview"
def pagina_overview():
    exibir_titulo_descricao()
    exibir_resumo_yaml()
    exibir_grafico_barras()
    exibir_insights_json()

# Função da página "Despesas"
def pagina_despesas():
    exibir_despesas()

# Função da página "Proposições"
def pagina_proposicoes():
    exibir_proposicoes()

# Função principal para gerenciar a navegação
def Main():
    st.sidebar.title("Navegação")

    # Verifica se já há uma página selecionada
    if "pagina_selecionada" not in st.session_state:
        st.session_state["pagina_selecionada"] = "Overview"

    # Botões de navegação
    if st.sidebar.button("Overview"):
        st.session_state["pagina_selecionada"] = "Overview"
    if st.sidebar.button("Despesas"):
        st.session_state["pagina_selecionada"] = "Despesas"
    if st.sidebar.button("Proposições"):
        st.session_state["pagina_selecionada"] = "Proposições"

    # Navegação condicional com base no estado da sessão
    if st.session_state["pagina_selecionada"] == "Overview":
        pagina_overview()
    elif st.session_state["pagina_selecionada"] == "Despesas":
        pagina_despesas()
    elif st.session_state["pagina_selecionada"] == "Proposições":
        pagina_proposicoes()

if __name__ == "__main__":
    Main()
