# Predição de aceitação e bloqueio em redes ópticas elásticas multi-núcleo

Repositório dos artefatos computacionais do estudo **Predição de aceitação e bloqueio de requisições em redes ópticas elásticas multi-núcleo utilizando técnicas de machine learning**.

O projeto investiga a classificação binária de requisições em redes ópticas elásticas com multiplexação por divisão espacial (SDM-EONs), distinguindo circuitos **aceitos** de circuitos **bloqueados** a partir de atributos relacionados ao estado da rede, ao espectro e à qualidade física da transmissão.

## Objetivo

Avaliar se modelos de aprendizado de máquina podem apoiar a triagem de requisições e antecipar a decisão de aceitação ou bloqueio de uma conexão. A proposta busca contribuir para:

- reduzir bloqueios indevidos;
- melhorar o aproveitamento de espectro e núcleos;
- apoiar decisões de alocação de recursos em cenários de tráfego dinâmico;
- comparar modelos baseados em árvores, probabilidade e grafos.

## Contexto do estudo

Nas SDM-EONs, uma requisição depende da combinação de rota, modulação, núcleo e espectro. A alocação precisa respeitar, entre outras condições, a continuidade e a contiguidade espectral, além dos requisitos de qualidade de transmissão (QoT). A fragmentação espectral, o ruído ASE, as interferências não lineares (NLI) e o crosstalk inter-núcleos (XT) podem provocar o bloqueio mesmo quando ainda existe capacidade livre.

O estudo transforma os códigos de resultado da simulação em uma variável binária:

| Valor | Interpretação |
| --- | --- |
| `1` | Requisição aceita |
| `0` | Requisição bloqueada, independentemente da causa |

## Estrutura do repositório

```text
sdm-eon-classifier-SOMMA/
├── base_de_dados/
│   └── baseMLJurandir.zip
├── notebook/
│	└── RedesElásticas_Somma.ipynb
└── README.md
```

### Artefatos

- [`base_de_dados/baseMLJurandir.csv`](base_de_dados/baseMLJurandir.zip): base de dados usada na análise, com 1.084.215 registros e 12 colunas antes do pré-processamento.
- [`notebook/RedesElásticas_Somma.ipynb`](notebook/RedesElásticas_Somma.ipynb): notebook com carregamento, exploração, engenharia de atributos, balanceamento, treinamento e avaliação.

## Dados

### Colunas originais

| Coluna | Descrição geral |
| --- | --- |
| `saltos` | Número de saltos da rota |
| `núcleo` | Núcleo selecionado na fibra multi-núcleo |
| `modulação` | Formato ou nível de modulação |
| `comprimento` | Comprimento da rota |
| `primeiro slot` | Primeiro slot espectral alocado |
| `ultimo slot` | Último slot espectral alocado |
| `XT` | Nível de crosstalk inter-núcleos |
| `OSNR` | Relação sinal-ruído óptico |
| `banda` | Largura de banda requerida |
| `consumo energia` | Variável removida antes do treinamento |
| `peso` | Variável removida antes do treinamento |
| `resultado` | Rótulo original da requisição |

O notebook corrige possíveis problemas de codificação nos nomes `núcleo` e `modulação`, converte `resultado` para a representação binária e cria atributos derivados. A feature `slots_usados` é calculada por:

```text
slots_usados = ultimo slot - primeiro slot + 1
```

Também são criadas `dist_por_salto`, `interacao_osnr_dist` e `densidade_espectral`. As variáveis numéricas selecionadas são escalonadas com `RobustScaler`, e a modulação é codificada com `LabelEncoder` quando está armazenada como texto.

## Fluxo metodológico

1. Importação das bibliotecas científicas e de aprendizado de máquina.
2. Carregamento da base em um `DataFrame` do pandas.
3. Correção dos nomes de colunas e binarização de `resultado`.
4. Criação e seleção de atributos derivados.
5. Análise exploratória, distribuição das classes e matriz de correlação.
6. Divisão estratificada em treino e teste, na proporção 70/30, com `random_state=42`.
7. Aplicação de undersampling somente no conjunto de treino.
8. Treinamento dos classificadores.
9. Avaliação no conjunto de teste sem balanceamento.
10. Comparação por métricas, matrizes de confusão e curvas ROC/AUC.

O balanceamento reduz a classe majoritária do conjunto de treino para o tamanho da classe minoritária. O conjunto de teste permanece com a distribuição original para representar melhor o cenário observado na prática.

## Modelos avaliados

### Random Forest

Conjunto de árvores de decisão treinadas sobre subconjuntos aleatórios dos dados e dos atributos. No notebook, é configurado com 200 árvores, profundidade máxima 15, `min_samples_leaf=5`, seleção `sqrt` de atributos e execução paralela.

### XGBoost

Modelo de gradient boosting em árvores, com treinamento sequencial e regularização. A configuração registrada usa 500 estimadores, taxa de aprendizado `0.03`, profundidade máxima 10, `subsample=0.8` e `colsample_bytree=0.7`.

### LightGBM

Implementação eficiente de gradient boosting baseada em árvores, treinada no notebook com `LGBMClassifier`, `random_state=42` e execução paralela.

### Naive Bayes

Classificador probabilístico baseado na hipótese de independência condicional entre atributos. O notebook aplica `PowerTransformer` com método Yeo-Johnson antes do treinamento do `GaussianNB`.

### Graph Convolutional Network (GCN)

Rede neural de grafos com três camadas convolucionais, normalização em lote, dropout e camada final de classificação. O treinamento utiliza AdamW por 100 épocas. No notebook disponível, o índice de arestas é construído com auto-conexões dos exemplos; portanto, a implementação não representa explicitamente a topologia física da rede.

## Métricas

São calculadas as métricas clássicas de classificação:

- **Acurácia**: proporção total de previsões corretas.
- **Precisão**: proporção das requisições previstas como aceitas que realmente foram aceitas.
- **Recall**: capacidade de identificar requisições aceitas.
- **F1-score**: média harmônica entre precisão e recall.
- **Matriz de confusão**: detalhamento de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.
- **ROC/AUC**: capacidade de separação entre as classes em diferentes limiares.

Neste problema, um falso negativo representa uma requisição aceita classificada como bloqueada, enquanto um falso positivo representa uma requisição bloqueada classificada como aceita.

## Resultados reportados no artigo

Os resultados abaixo são os apresentados no manuscrito e servem como referência do estudo:

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-score (%) |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 80,67 | 89,37 | 79,96 | 84,40 |
| LightGBM | 80,46 | 89,05 | 79,96 | 84,26 |
| Random Forest | 80,26 | 89,22 | 79,42 | 84,03 |
| GCN | 74,14 | 90,74 | 67,34 | 77,31 |
| Naive Bayes | 70,00 | 85,52 | 65,18 | 73,98 |

O XGBoost apresentou o melhor desempenho global segundo acurácia e F1-score. LightGBM e Random Forest tiveram resultados próximos, enquanto GCN e Naive Bayes apresentaram menor desempenho geral. A GCN obteve a maior precisão, mas com recall inferior, indicando uma postura mais restritiva na classificação das requisições aceitas.

## Como executar

### Pré-requisitos

- Python 3.9 ou superior;
- Jupyter Notebook ou VS Code com a extensão Jupyter;
- ambiente com suporte a bibliotecas científicas;
- PyTorch e PyTorch Geometric para executar a GCN.

### Instalação das dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm torch torch-geometric
```

### Execução

Abra [`notebook/RedesElásticas_Somma.ipynb`](notebook/RedesElásticas_Somma.ipynb) em um ambiente Jupyter e execute as células em ordem.

O notebook foi escrito originalmente esperando o arquivo CSV no diretório de trabalho atual. A partir da raiz deste repositório, ajuste o carregamento para:

```python
df = pd.read_csv('base_de_dados/baseMLJurandir.csv')
```

Como alternativa, execute o notebook com `base_de_dados` como diretório de trabalho e mantenha o caminho `baseMLJurandir.csv`.

## Reprodutibilidade e observações

- A divisão treino/teste e o undersampling usam `random_state=42`.
- Os resultados da tabela acima são os reportados no artigo; o notebook distribuído não possui execução persistida nas células.
- Antes de uma execução do zero, verifique as variáveis de predição usadas nas células de avaliação e comparação da GCN e do Naive Bayes, pois elas dependem da execução completa das etapas anteriores.
- O CSV versionado contém as colunas físicas disponíveis no repositório. A descrição do artigo menciona atributos adicionais derivados ou obtidos na simulação; neste repositório, parte deles é reconstruída pelo notebook a partir das colunas originais.
- O balanceamento ocorre somente no treino, evitando alterar a distribuição natural do teste.

## Relação com o artigo

O trabalho aborda redes ópticas elásticas multi-núcleo, alocação RMSCA, fragmentação espectral, QoT, OSNR, NLI e crosstalk inter-núcleos, além da aplicação de Random Forest, Naive Bayes, XGBoost, LightGBM e GCN. O notebook e a base deste repositório concentram os procedimentos computacionais de preparação dos dados, treinamento e avaliação dos modelos.

## Licença e uso dos dados

Este repositório não declara uma licença de software ou de dados. Consulte os autores antes de redistribuir a base, os resultados ou os artefatos do estudo.
