import requests
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import json



def deputados():
    url_base = 'https://dadosabertos.camara.leg.br/api/v2/'
    endpoint = 'deputados'
    url = url_base + endpoint

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        deputados = data['dados'] 

        df = pd.DataFrame(deputados)

        output_path = 'data/deputados.parquet'
        df.to_parquet(output_path, index=False)
        print(f"Arquivo salvo com sucesso em: {output_path}")
    else:
        print(f"Erro na requisição. Status code: {response.status_code}")




def llm_insights():
    """
    Explicação do objetivo de cada elemento na função e análise da resposta do LLM:

    Esta função realiza o seguinte processo:

    1. **Obtenção dos dados**:
        - Leitura de arquivo parquet utilizando pandas para obter os dados.

    2. **Criação do Código para Análise de Dados**:
       - Utiliza um modelo LLM (ChatGoogleGenerativeAI) para gerar um código Python que, com base em um DataFrame de dados de deputados, cria um gráfico de pizza exibindo a distribuição percentual de deputados por partido.
       - O código gerado pelo LLM é extraído e preparado para execução, sem a necessidade de recarregar os dados do DataFrame, pois ele já foi previamente carregado.

    3. **Execução do Código Gerado**:
       - O código gerado pelo LLM é executado via a função `exec()`, o que permite que o gráfico de pizza seja gerado e salvo no diretório especificado.
       - Caso ocorra um erro durante a execução do código, ele será capturado e impresso.

    4. **Solicitação de Insights ao LLM**:
       - Após a execução do código, o LLM é consultado novamente com um prompt que utiliza o código gerado para pedir uma análise política detalhada sobre a distribuição de deputados por partido, incluindo:
         1. Identificação dos partidos com maior influência na Câmara.
         2. Avaliação de como a distribuição pode afetar a dinâmica política.
         3. Identificação de tendências ou padrões no equilíbrio de poder entre os partidos.

    5. **Armazenamento dos Insights**:
       - A resposta do LLM, que contém os insights sobre a distribuição política, é armazenada em um arquivo JSON localizado em `data/insights_distribuicao_deputados.json`.

    6. **Tratamento de erros**:
        - Se ocorrer algum erro ao executar o código gerado ou durante a solicitação de insights, uma mensagem de erro será exibida.

    Análise dos insights obtidos pelo LLM:
    O LLM foi capaz de identificar corretamente os partidos com maior influência dentro da Câmara, o que é um ponto positivo. No entanto, 
    houve uma discrepância entre os valores apresentados nos insights gerados pelo modelo e os valores obtidos por meio do código ou visualizados 
    no gráfico (docs/distribuicao_deputados.png), o que pode gerar confusão. Além disso, ao tratar das implicações políticas, o modelo apresentou 
    repetição de algumas respostas, o que comprometeu a fluidez e a clareza da análise, tornando-a menos impactante. Apesar dessas limitações, 
    a análise fornecida ainda consegue representar de maneira eficaz a realidade da distribuição de poder na Câmara, destacando as consequências 
    de um cenário em que poucos partidos possuem uma representatividade dominante, o que pode afetar significativamente as dinâmicas políticas 
    e as decisões estratégicas dentro do câmara dos deputados.

    """

    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyDUm58IAr5Ufp6kTw-HWRKnIoU0hBBI-qc")

    df = pd.read_parquet("data/deputados.parquet")

    prompt_1 = f"""
    Você é um Especialista em programação e análise de dados em Python, você está trabalhando com uma base de dados referente aos deputados no
    Brasil. 
    Irei te passar o DataFrame com todas as informações referente ao deputados e preciso que você crie um código Python pronto para ser executado sem
    nenhum tipo de explicação apenas o script funcional, que atenda os seguintes critérios:
        - Gerar um gráfico de pizza com o total e o percentual de deputados de cada partido, 
        - Resultado deve ser salvo em 'docs/distribuicao_deputados.png
    Dados:
    {df}

    Na geração do código não precisa carregar o DataFrame pois já realizo a leitura do mesmo previamente, quero apenas o código que realiza a análise.
    """

    resposta = llm.invoke(prompt_1).content
    code = resposta.replace('```python', '').replace('```', '').strip()

    try:
        exec(code)

        prompt_2 = f"""
        Você é um especialista em análise política e distribuição de poder no Brasil. Abaixo está o código em Python que obtém a distribuição percentual 
        de deputados por partido na Câmara dos Deputados:
        {code}

        Com base nessa distribuição, forneça insights sobre:
        1. Quais os três partidos têm mais influência na Câmara?
        2. Como a distribuição de partidos pode afetar a dinâmica das votações e decisões políticas?
        3. Existe alguma tendência que pode ser observada em relação ao equilíbrio de poder entre partidos?

        Utilize esses dados para fornecer uma análise detalhada que ajude a entender as implicações políticas dessa distribuição.

        A resposta deve incluir uma análise aprofundada sobre a influência dos partidos, com base na distribuição atual e possíveis cenários futuros.
        """

        resposta = llm.invoke(prompt_2).content

        insights = {
            "insights": resposta
        }

        json_data = json.dumps(insights, ensure_ascii=False, indent=4)
        with open("data/insights_distribuicao_deputados.json", "w") as f:
            f.write(json_data)

    except Exception as e:
        print(f"Ocorreu um erro ao executar o código: {e}")




def deputados_despesas():

    df = pd.read_parquet("data/deputados.parquet")
    lista_ids = df["id"].to_list()

    url_base = 'https://dadosabertos.camara.leg.br/api/v2'

    todas_despesas = []
    for deputado_id in lista_ids:
        endpoint = f"/deputados/{deputado_id}/despesas?ano=2024&mes=11"
        url = f"{url_base}{endpoint}"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                print(data)

                data = response.json()
                despesas = data.get('dados', [])

                for despesa in despesas:
                    despesa["idDeputado"] = deputado_id

                todas_despesas.extend(despesas)

        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição para deputado {deputado_id}: {e}")
            continue



    df_despesas = pd.DataFrame(todas_despesas)
    df_despesas.groupby(["dataDocumento", "idDeputado", "tipoDespesa"], as_index=False
    ).agg({
        "valorDocumento": "sum",
        "valorLiquido": "sum"
    })

    output_path = "data/despesas_deputados.parquet"
    df_despesas.to_parquet(output_path, index=False)
    print(f"Arquivo consolidado salvo com sucesso em: {output_path}")


def deputados_despesas_insights():

    df = pd.read_parquet("data/despesas_deputados.parquet")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyDUm58IAr5Ufp6kTw-HWRKnIoU0hBBI-qc")

    prompt_1 = f"""
    Você é um especialista em programação e análise de dados em Python. Você está trabalhando com uma base de dados referente aos deputados no Brasil.
    Irei te passar o DataFrame com todas as informações sobre os deputados e suas despesas. Preciso que você crie um código Python funcional e pronto para execução que atenda aos três critérios abaixo:
    
    1. Retorne o ID do deputado que mais gastou.
    2. Retorne o tipo de despesa que possui maior valor total.
    3. Retorne o tipo de documento com maior valor total **e o valor correspondente**.

    O código deve:
    - Utilizar apenas `groupby` e `sum` para realizar as análises.
    - Retornar as respostas de forma estruturada no seguinte formato:
      ```python
      respostas = {{
          "deputado_mais_gastou": <ID do deputado>,
          "tipo_despesa_maior_valor": <Tipo de despesa>,
          "tipo_documento_maior_valor": {{
              "documento": <Tipo de documento>,
              "valor": <Valor total>
          }}
      }}
      ```
    - Imprimir as respostas no console no mesmo formato acima.

    Importante:
    - Não é necessário incluir a leitura do DataFrame no código.
    - O código deve ser funcional e direto, sem explicações ou comentários desnecessários.
    Dados:
    {df}
    """

    resposta = llm.invoke(prompt_1).content
    code = resposta.replace('```python', '').replace('```', '').strip()
    
    try:
        local_vars = {}
        exec(code, {"df": df}, local_vars)

        respostas = local_vars.get("respostas", {})
        deputado_mais_gastou = respostas.get("deputado_mais_gastou", None)
        tipo_despesa_maior_valor = respostas.get("tipo_despesa_maior_valor", None)
        tipo_documento_maior_valor = respostas.get("tipo_documento_maior_valor", None)

        prompt_2 = f"""
        Você é um especialista em análise política e distribuição de poder no Brasil. Abaixo estão os resultados de uma análise sobre as despesas dos deputados:
            1. O ID do deputado que mais gastou: {deputado_mais_gastou}
            2. O tipo de despesa com maior valor total: {tipo_despesa_maior_valor}
            3. O tipo de documento com maior valor total e o valor correspondente: {tipo_documento_maior_valor}

        Com base nessas informações, forneça insights valiosos e detalhados sobre as implicações políticas das despesas dos deputados no Brasil.
        A resposta deve incluir uma análise aprofundada.
        """

        resposta = llm.invoke(prompt_2).content

        insights = {
            "insights": resposta
        }

        json_data = json.dumps(insights, ensure_ascii=False, indent=4)
        with open("data/insights_despesas_deputados.json", "w") as f:
            f.write(json_data)

    except Exception as e:
        print(f"Ocorreu um erro ao executar o código: {e}")



def coletar_proposicoes():

    url_base = "https://dadosabertos.camara.leg.br/api/v2/proposicoes"
    headers = {
        "Accept": "application/json"
    }
    output_path = 'data/proposicoes_deputados.parquet'
    dataInicio = '2020-01-01'
    dataFim = '2024-11-30'
    temas = [40, 46, 62]
    total_por_tema = 10
    
    todas_proposicoes = []

    for tema in temas:
        params = {
            "dataInicio": dataInicio,
            "dataFim": dataFim,
            "codTema": tema,
            "itens": total_por_tema
        }
        
        response = requests.get(url_base, headers=headers, params=params)
        
        if response.status_code == 200:
            dados = response.json().get("dados", [])
            print(dados)
            for proposicao in dados:
                proposicao["codTema"] = tema
                todas_proposicoes.append(proposicao)
                print(todas_proposicoes)
        else:
            print(f"Erro ao coletar proposições para o tema")
    
    if todas_proposicoes:
        df = pd.DataFrame(todas_proposicoes)
        df.to_parquet(output_path, index=False)
        print(f"Arquivo salvo com sucesso em: {output_path}")
    else:
        print("Nenhuma proposição foi coletada.")




def sumarizacao_preposicoes():

    model_name = 'gemini-pro'
    google_api_key = "AIzaSyDUm58IAr5Ufp6kTw-HWRKnIoU0hBBI-qc"
    window_size = 100
    overlap_size = 25

    df = pd.read_parquet("data/proposicoes_deputados.parquet")

    textos = df["ementa"].dropna().tolist()

    def split_into_chunks(text_list, n, m):
        return [text_list[i:i + n] for i in range(0, len(text_list), n - m)]

    chunks = split_into_chunks(textos, window_size, overlap_size)


    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        system_instruction="You are a summarizer for text chunks.",
        generation_config={
            'temperature': 0.2,
            'top_p': 0.8,
            'max_output_tokens': 500,
        }
    )

    def summarize_chunk(chunk):
        prompt = f"Summarize the following text:\n{chunk}"
        response = model.invoke(prompt)
        return response.content

    print("Resumindo chunks...")
    chunk_summaries = [summarize_chunk(chunk) for chunk in chunks]

    final_prompt = f"Summarize the following summaries:\n{' '.join(chunk_summaries)}"
    print("Criando resumo final...")
    final_response = model.invoke(final_prompt)

    print(final_response.content)

    try:
        json_data = json.dumps(final_response.content, ensure_ascii=False, indent=4)
        with open("data/sumarizacao_proposicoes.json", "w") as f:
            f.write(json_data)
    except:
        print('Erro ao salvar aquivo JSON')
 


def main():
    """Função principal para gerenciar a execução do programa."""
    #deputados()
    #llm_insights()
    #deputados_despesas()
    #deputados_despesas_insights()
    #coletar_proposicoes()
    #sumarizacao_preposicoes()

if __name__ == "__main__":
    main()

