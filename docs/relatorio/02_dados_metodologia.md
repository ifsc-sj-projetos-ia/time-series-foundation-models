**2. Dados e Metodologia**

**2.1 Conjunto de Dados**

O *dataset* utilizado é o *Predict Energy Behavior of Prosumers*, disponibilizado pela Enefit, concessionária de energia elétrica da Estônia, hospedado na plataforma Kaggle. Os dados compreendem registros horários de produção e consumo de energia agregados por unidade de previsão, definida pela combinação de condado (15 condados estonianos mais uma categoria desconhecida), tipo de consumidor (residencial ou comercial) e tipo de produto energético (quatro modalidades contratuais), totalizando 69 unidades de previsão.

Para cada unidade, há dois registros por instante: um para energia consumida da rede (*is_consumption=1*) e outro para energia injetada na rede (*is_consumption=0*), doravante denominados *target_import* e *target_export*, respectivamente. O período coberto vai de setembro de 2021 a maio de 2023 (aproximadamente 21 meses), totalizando cerca de 15.300 horas por unidade.

As bases auxiliares incluem:
- **client.csv**: cadastro de prosumers por unidade, contendo número de instalações (*eic_count*) e capacidade instalada (*installed_capacity*), em escala diária.
- **historical_weather.csv**: observações meteorológicas históricas de 112 estações, com temperatura, ponto de orvalho, precipitação, nebulosidade, velocidade do vento e radiação solar.
- **forecast_weather.csv**: previsões meteorológicas com horários de origem e horizontes de 1 a 48 horas à frente.
- **electricity_prices.csv**: preços horários do mercado de energia elétrica do dia seguinte (*day-ahead*), em EUR/MWh.

**2.2 Pré-processamento**

O pré-processamento foi realizado localmente em máquina com 8 GB de RAM e GPU GTX 1650, processando uma unidade de previsão por vez para evitar estouro de memória.

Para cada unidade, as seguintes etapas foram executadas:

1. **Pivot** dos registros de *is_consumption* em duas colunas: *target_export* e *target_import*.
2. **Junção** com *client.csv* pelas chaves (condado, tipo, data), expandindo capacidade instalada e contagem de instalações como séries temporais.
3. **Atribuição meteorológica**: cada um dos 112 pontos de observação foi associado ao condado mais próximo por distância de Haversine, e as leituras foram agregadas por condado por média aritmética.
4. **Junção** com *electricity_prices* pela data/hora.
5. **Normalização**: aplicação de *StandardScaler* (média zero, variância unitária) independente por canal, ajustado exclusivamente no conjunto de treino e aplicado aos conjuntos de validação e teste.
6. **Exportação** em dois formatos: CSV (inspeção visual e auditoria) e arquivo PyTorch (.pt) com tensores contínuos, parâmetros do escalonador e metadados.

A divisão temporal dos dados segue os blocos do próprio *dataset*:
- **Treino**: blocos 0 a 600 (~94% dos dados, ~20 meses)
- **Validação**: blocos 601 a 633 (~5% dos dados, ~1 mês)
- **Teste**: blocos 634 a 637 (~1% dos dados, 4 dias, 96 horas)

**2.3 Modelos**

Seis configurações foram avaliadas:

1. **Persistence (linha de base)**: repete a última observação disponível para todo o horizonte de previsão.
2. **Seasonal Naive (linha de base)**: repete o ciclo das últimas 24 horas, capturando a sazonalidade diária.
3. **TTM2 zero-shot**: TinyTimeMixer v2, revisão *512-96-ft-r2.1*, 805 mil parâmetros, contexto de 512 *timesteps*, previsão de 96 *timesteps*, sem ajuste fino.
4. **TTM2 *fine-tuned* (stride=1)**: mesma revisão, treinado por 10 épocas com 100% dos dados e *backbone* descongelado, utilizando janelas deslizantes densas (passo 1).
5. **FlowState r1.0**: SSM com decodificador funcional contínuo, 9 milhões de parâmetros, contexto de 2048 *timesteps* (nativo), previsão de 96 *timesteps*, modo *zero-shot*.
6. **FlowState r1.1**: mesma arquitetura com 18,5 milhões de parâmetros, contexto de 4096 *timesteps*.

Todos os modelos foram executados localmente na GPU GTX 1650 (3,9 GB VRAM). Nenhum dos experimentos exigiu infraestrutura em nuvem.

**2.4 Métricas de Avaliação**

As métricas foram calculadas separadamente por unidade de previsão e por variável-alvo, mantendo a granularidade completa dos dados. A agregação nacional (soma das 69 unidades) é fornecida como tabela suplementar para comparação com *benchmarks* anteriores.

O SMAPE é adotado como métrica primária por sua robustez a valores próximos de zero, frequentes na geração solar noturna, que tornam o MAPE instável e dominado por denominadores muito pequenos.
