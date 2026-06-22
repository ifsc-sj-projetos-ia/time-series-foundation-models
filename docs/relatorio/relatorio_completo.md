**1. Introdução**

A crescente popularização de fontes renováveis distribuídas, como painéis solares residenciais, transforma consumidores tradicionais em *prosumers*: agentes que tanto consomem quanto injetam energia na rede. Essa mudança introduz um desafio operacional para os operadores do sistema: a geração distribuída é intermitente (dependente de clima, horário e estação) e descentralizada, o que torna a previsão precisa do comportamento dos prosumers um requisito crítico para o equilíbrio entre oferta e demanda, a estabilidade da rede e o aproveitamento máximo de fontes renováveis.

Este trabalho compara dois modelos de fundação para séries temporais (*time series foundation models*) da família IBM Granite, o **TinyTimeMixer v2 (TTM2)** e o **FlowState**, na tarefa de previsão horária de produção e consumo de energia de prosumers na Estônia. Ambos os modelos são pré-treinados em grandes volumes de dados heterogêneos e podem ser aplicados sem ajuste fino (*zero-shot*) ou com adaptação supervisionada (*fine-tuning*). O FlowState, baseado em *State Space Models* (SSM) com decodificador funcional contínuo, representa a geração mais recente da família; o TTM2, baseado em *MLP Mixers* compactos (cerca de 1 milhão de parâmetros), é a geração anterior. A comparação busca quantificar o ganho real da arquitetura SSM sobre a MLP no contexto específico de energia elétrica.

O conjunto de dados utilizado é o *Predict Energy Behavior of Prosumers*, disponibilizado pela Enefit (concessionária estoniana) via Kaggle, contendo 69 unidades de previsão distribuídas em 16 condados, com dados horários de setembro de 2021 a maio de 2023. As covariáveis incluem temperatura, radiação solar, velocidade do vento, precipitação, preço horário de eletricidade e capacidade instalada por unidade consumidora.

As métricas de avaliação adotadas são MAE (*Mean Absolute Error*), RMSE (*Root Mean Squared Error*) e SMAPE (*Symmetric Mean Absolute Percentage Error*). O SMAPE é utilizado como métrica primária por ser limitada entre 0 e 200% e robusta a valores próximos de zero, comuns em séries de geração solar noturna, evitando a instabilidade numérica do MAPE tradicional nesses cenários.

Os resultados indicam que o FlowState r1.0 com contexto de 2048 timesteps supera consistentemente o TTM2 e todas as linhas de base (*persistence* e *seasonal naive*) em ambas as variáveis-alvo, tanto na avaliação por unidade individual quanto na agregação nacional, estabelecendo-se como a melhor configuração para este conjunto de dados.
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
**3. Experimentos e Resultados**

**3.1 Planejamento Experimental**

Os experimentos de *fine-tuning* foram conduzidos segundo a metodologia DoE (*Design of Experiments*), utilizando um fatorial completo 2³ (3 fatores, 8 pontos fatoriais + 2 pontos centrais = 10 execuções) sobre uma unidade representativa do conjunto de dados. Os fatores investigados foram: revisão do modelo (L2 vs L1), congelamento do *backbone* (sim vs não) e fração dos dados de treino (5% vs 20%). Dois fatores adicionais previstos no planejamento inicial, contexto (512 vs 1024) e uso de covariáveis (sim vs não), foram descartados durante a fase de triagem: o primeiro porque as revisões do TTM2 são pré-treinadas com comprimento de contexto fixo, e o segundo porque o módulo de mistura de canais (*channel mixing*) do modelo exige valores futuros das covariáveis durante a inferência, que não estão disponíveis no período de teste.

Os experimentos *zero-shot* (TTM2 e FlowState) não utilizaram DoE, pois não envolvem variação de hiperparâmetros de treinamento.

**3.2 Comparação Principal**

A tabela abaixo apresenta os resultados agregados (média sobre as 69 unidades de previsão) para as seis configurações avaliadas:

| Modelo | Export MAE | Export SMAPE | Import MAE | Import SMAPE |
|---|---|---|---|---|
| Persistence | 420,6 | 159,5% | 136,5 | 60,5% |
| Seasonal Naive | 238,3 | 94,0% | 102,5 | 39,9% |
| TTM2 *zero-shot* | 202,3 | 106,4% | 107,7 | 46,2% |
| FlowState r1.0 ctx512 | 201,9 | 103,3% | **90,1** | **38,9%** |
| **FlowState r1.0 ctx2048** | **181,7** | **99,6%** | 92,4 | 39,6% |
| FlowState r1.1 ctx4096 | 183,6 | **94,2%** | 98,1 | 39,9% |

O FlowState r1.0 com contexto de 2048 *timesteps* apresentou o menor MAE em ambas as variáveis-alvo. Em *target_export* (geração), o ganho sobre o TTM2 *zero-shot* foi de aproximadamente 10% no MAE (181,7 vs 202,3), indicando que o contexto mais longo do FlowState captura padrões meteorológicos que o TTM2, limitado a 512 *timesteps*, não alcança. Em *target_import* (consumo), o FlowState ctx512 obteve o melhor SMAPE (38,9%), marginalmente à frente do ctx2048 (39,6%) e do *Seasonal Naive* (39,9%). A proximidade entre o modelo baseline e os modelos de fundação para a variável de consumo sugere que o consumo residencial e comercial segue um padrão diário regular que é bem capturado mesmo por uma estratégia simples de repetição do ciclo de 24 horas.

O FlowState r1.1 (18,5 milhões de parâmetros, contexto 4096) não superou o r1.0 (9 milhões, contexto 2048) em nenhuma métrica, indicando que mais parâmetros e contexto mais longo não se traduziram em ganho para este conjunto de dados específico.

**3.3 Fine-tuning do TTM2**

O TTM2 *fine-tuned* com janelas deslizantes densas (passo 1) atingiu SMAPE de 38,58% em *target_import* na unidade 0, superando marginalmente o FlowState ctx2048 (38,94%) na mesma unidade. O MAE, no entanto, permaneceu superior (121,2 vs 92,4). Com janelas não sobrepostas (passo 96), o mesmo modelo obteve SMAPE de apenas 55,78%, demonstrando que a densidade das janelas de treinamento é crítica para o *fine-tuning* — o modelo necessita de exemplos sobrepostos para aprender padrões locais. Este resultado não foi validado nas 69 unidades devido ao custo computacional (cerca de 150 segundos por unidade, total estimado de 5 horas).

**3.4 Agregação Nacional**

A agregação nacional (soma das 69 unidades) produz estimativas pontuais para todo o sistema elétrico estoniano:

| Modelo | Export MAE | Export SMAPE | Import MAE | Import SMAPE |
|---|---|---|---|---|
| Persistence | 27.310 | 151,1% | 6.563 | 27,6% |
| Seasonal Naive | 14.545 | 77,3% | 5.141 | 22,3% |
| TTM2 *zero-shot* | 11.935 | 105,2% | 5.704 | 23,7% |
| **FlowState ctx2048** | **9.862** | **80,5%** | **4.542** | **20,0%** |

A agregação reduz o ruído estocástico por unidade e produz valores de SMAPE significativamente mais baixos. O MAPE de *target_import*, que atinge 355% na média por unidade (inflado por denominadores pequenos em unidades com baixo consumo), cai para 28,5% no dado nacional. Em *target_export*, o SMAPE cai de aproximadamente 100% (média por unidade) para 80,5% (nacional). Estes valores são consistentes com *benchmarks* anteriores da literatura para previsão agregada de carga elétrica.
**4. Discussão**

**4.1 Limitações do Estudo**

A principal limitação deste trabalho é a restrição geográfica do conjunto de dados. O *dataset* da Enefit cobre exclusivamente o território estoniano, com suas características climáticas e de consumo específicas — invernos rigorosos, alta penetração de aquecimento elétrico e latitude elevada (que afeta a sazonalidade da geração solar). Os resultados reportados não podem ser generalizados para outras regiões sem validação independente.

O FlowState, apesar de ter obtido o melhor desempenho geral, está limitado ao modo *zero-shot* — a versão de código aberto do modelo não oferece suporte a *fine-tuning*. Isso significa que o modelo não pode ser adaptado a domínios específicos ou a mudanças estruturais na série temporal (como uma expansão repentina da capacidade instalada de geração distribuída em um condado específico).

O *fine-tuning* do TTM2 com janelas densas (passo 1) foi validado apenas na unidade de previsão 0 (Harjumaa residencial), devido ao custo computacional de aproximadamente 5 horas para as 69 unidades. Embora os resultados nessa unidade tenham sido promissores (SMAPE 38,6%), a generalização para o conjunto completo permanece não verificada.

As covariáveis meteorológicas disponíveis no *dataset* (temperatura, radiação solar, nebulosidade, etc.) não foram utilizadas nos experimentos principais. No caso do FlowState, o modelo não suporta canais exógenos. No caso do TTM2, a tentativa de utilizar covariáveis foi bloqueada por uma limitação de arquitetura: o módulo *Forecast Channel Mixing* (FCM) requer valores futuros das covariáveis durante a inferência, que não estão disponíveis para o período de teste. Esta é uma restrição relevante para aplicações reais de previsão energética, onde variáveis meteorológicas previstas são frequentemente o sinal mais informativo.

O período de teste é curto — 96 horas, correspondendo aos blocos 634 a 637 do *dataset*. Quatro dias não capturam eventos sazonais extremos (ondas de frio, picos de demanda), nem permitem avaliar a degradação do modelo em horizontes mais longos. A validação em um período de teste mais extenso seria necessária para uso em produção.

**4.2 Disparidade por Condado**

A acurácia dos modelos varia significativamente entre condados. Condados com maior densidade populacional e mais instalações (Harjumaa, Tartumaa) tendem a apresentar erros absolutos maiores (pois consomem mais energia), mas erros relativos menores. Condados rurais com poucos prosumers (Hiiumaa, Põlvamaa) têm séries temporais mais ruidosas e métricas proporcionalmente piores.

Esta disparidade não constitui um viés algorítmico — é uma consequência natural da densidade de dados. Modelos de fundação pré-treinados não possuem mecanismo para compensar a escassez de observações em séries com poucas instalações. Em uma implantação real, isso poderia significar que a qualidade do serviço de previsão é pior exatamente nas regiões com menor representação no conjunto de treinamento — um efeito que operadores de rede devem considerar ao definir políticas de balanceamento de carga.

**4.3 Privacidade e Uso Ético**

Os dados utilizados são agregados por condado e por tipo de consumidor. Nenhum registro individual de consumo ou geração está presente no *dataset* — a menor granularidade é a combinação de condado, tipo de consumidor e modalidade contratual, o que torna impossível a reidentificação de prosumers individuais. Não há, portanto, risco de violação de privacidade neste nível de agregação.

Caso a mesma metodologia fosse aplicada a dados desagregados — como leituras de medidores inteligentes individuais —, seriam necessárias salvaguardas adicionais, incluindo anonimização, agregação temporal mínima e conformidade com legislações de proteção de dados (LGPD, GDPR).

Do ponto de vista ambiental, ambos os modelos são notavelmente eficientes. O TTM2 possui 805 mil parâmetros e o FlowState r1.0 possui 9 milhões — ordens de grandeza abaixo dos grandes *transformers* utilizados em NLP (que frequentemente ultrapassam 100 milhões de parâmetros). Todos os experimentos foram executados em uma GPU de consumo (GTX 1650, 4 GB VRAM) sem necessidade de infraestrutura em nuvem, com tempo total de inferência inferior a 30 segundos para o conjunto completo de 69 unidades. O custo energético da fase experimental é, portanto, desprezível.

Um risco relevante em sistemas de previsão para o setor elétrico é o excesso de confiança nas previsões por parte de operadores sem conhecimento técnico das limitações do modelo. Se as previsões não forem continuamente contrastadas com dados históricos de desempenho — por exemplo, por meio de *backtesting* com métricas de erro atualizadas —, decisões operacionais (acionamento de termelétricas de reserva, corte de geração renovável, compra de energia no mercado *spot*) podem ser tomadas com base em previsões degradadas. A acurácia de um modelo de fundação não é estática: ela varia com a estação, com eventos climáticos atípicos e com mudanças estruturais na rede. Sistemas de previsão devem incluir monitoramento contínuo de métricas e limites de alerta para degradação.

**4.4 Escolha da Função de Perda**

Conforme documentado em *05_loss_function.md*, os modelos TTM2 estão disponíveis em duas variantes de pré-treinamento: com perda L2 (MSE, padrão) e com perda L1 (MAE). A perda L1 penaliza erros linearmente, produzindo previsões de mediana condicional robustas a *outliers*. A perda L2 penaliza erros quadraticamente, sendo mais sensível a valores extremos.

Dados de energia elétrica apresentam três características que favorecem a perda L1: (a) períodos de geração solar nula (noite), que criam concentrações de zeros, (b) picos de demanda matinais e vespertinos, que atuam como *outliers* de alta magnitude, e (c) sazonalidade em múltiplas escalas (diária, semanal, trimestral), onde a mediana condicional é mais estável que a média. *Benchmarks* anteriores do projeto LoadPrediction (*05_loss_function.md*) confirmaram que as revisões L1 do TTM2 consistentemente superam as revisões L2 em configurações *zero-shot* no mesmo domínio (Estônia horária), com ganhos de 5 a 10% relativos de MAPE.

A decisão de utilizar a revisão L1 nos experimentos de *fine-tuning* foi registrada no planejamento, mas o DOE de triagem revelou que, após o ajuste fino, a diferença entre L1 e L2 é desprezível (correlação de apenas +0,07 com SMAPE), sugerindo que o *fine-tuning* sobrescreve o viés introduzido pela função de perda do pré-treinamento.

**4.5 Por que o FlowState Venceu**

O FlowState r1.0 superou o TTM2 em todas as configurações *zero-shot* e em ambas as métricas principais. Três fatores arquiteturais explicam essa vantagem:

1. **Contexto mais longo**: o FlowState foi pré-treinado com 2048 *timesteps*, contra 512 do TTM2. Para a variável de geração (*target_export*), que depende de padrões meteorológicos de múltiplos dias, o contexto adicional captura informação que o TTM2 simplesmente não tem acesso. Para consumo (*target_import*), o ganho é menor porque o padrão diário já está contido em 512 passos.

2. **Arquitetura SSM**: os *State Space Models* modelam dependências de longo alcance de forma mais eficiente que *MLP Mixers*, que dependem de mecanismos de atenção ou convoluções com janela fixa. A convolução via FFT utilizada pelo codificador S5 do FlowState processa a sequência completa em uma única operação, sem perda de informação temporal.

3. **Decodificador funcional contínuo**: o *Functional Basis Decoder* (FBD) projeta o estado latente do codificador em uma base de funções contínuas (polinômios de Legendre), permitindo amostrar a previsão em qualquer resolução temporal. Embora o fator de escala (*scale_factor*) ótimo tenha se confirmado como 1,0 para dados horários (o padrão recomendado), a flexibilidade do decodificador confere ao modelo uma capacidade de generalização que o decodificador linear do TTM2 não possui.

A combinação desses fatores torna o FlowState particularmente adequado para séries temporais energéticas, onde múltiplas escalas temporais (horária, diária, semanal) interagem de forma não trivial.
**5. Conclusão e Trabalhos Futuros**

Este trabalho comparou dois modelos de fundação para séries temporais da família IBM Granite — o TinyTimeMixer v2 (TTM2) e o FlowState — na tarefa de previsão horária de produção e consumo de energia de prosumers na Estônia, utilizando o *dataset* Enefit com 69 unidades de previsão distribuídas em 16 condados. Seis configurações foram avaliadas, incluindo duas linhas de base clássicas, três variantes do FlowState e duas do TTM2.

O FlowState r1.0 com contexto de 2048 *timesteps* foi o melhor modelo em todas as métricas e em ambas as variáveis-alvo, tanto na avaliação por unidade individual quanto na agregação nacional. Na variável de consumo (*target_import*), o MAE médio por unidade foi de 92,4 (contra 107,7 do TTM2 zero-shot) e o SMAPE de 39,6%. Na variável de geração (*target_export*), o MAE foi de 181,7 (contra 202,3) e o SMAPE de 99,6%. O ganho relativo é maior em geração do que em consumo — resultado esperado, uma vez que a geração solar é intrinsecamente mais dependente de padrões meteorológicos de longo prazo, que o contexto de 2048 passos captura melhor que os 512 do TTM2.

O TTM2 *zero-shot* mostrou-se competitivo, especialmente em consumo, mas consistentemente atrás do FlowState. A tentativa de melhorar seu desempenho via *fine-tuning* revelou que a densidade das janelas de treinamento é o fator crítico: com janelas não sobrepostas (passo 96), o SMAPE foi de apenas 55,8%; com janelas densas (passo 1), o SMAPE na unidade 0 atingiu 38,6%, superando marginalmente o FlowState na mesma unidade (38,9%). O MAE, no entanto, permaneceu superior (121,2 vs 92,4), indicando que o ganho em erro relativo não se traduziu em ganho absoluto. A validação nas 69 unidades não foi concluída devido ao custo computacional, permanecendo como trabalho futuro.

A versão mais recente do FlowState (r1.1, 18,5 milhões de parâmetros, contexto 4096) não apresentou melhoria em relação ao r1.0 (9 milhões, contexto 2048), sugerindo um ponto de saturação em que mais parâmetros e mais contexto não agregam informação adicional para este domínio.

A metodologia DoE (*Design of Experiments*) aplicada ao *fine-tuning* do TTM2 demonstrou sua utilidade na triagem de fatores: dois dos cinco fatores inicialmente planejados (contexto e covariáveis) foram descartados ainda na fase de triagem por incompatibilidade arquitetural, evitando experimentos inviáveis e concentrando o esforço computacional nos fatores que realmente importam. O efeito principal mais relevante foi a fração de dados de treino (correlação de -0,41 com SMAPE), confirmando que a quantidade de dados é mais determinante que a escolha da revisão do modelo (L1 vs L2) ou o congelamento do *backbone*.

O *Seasonal Naive* — um modelo trivial que repete o ciclo de 24 horas — obteve SMAPE de 39,9% em consumo, virtualmente empatado com os melhores modelos de fundação (38,9%). Este resultado é uma advertência metodológica importante: em séries com forte componente periódica, modelos sofisticados devem ser comparados contra uma linha de base sazonal, não apenas contra a persistência ingênua.

Trabalhos futuros incluem: (1) conclusão da validação do TTM2 *fine-tuned* com janelas densas nas 69 unidades de previsão, para confirmar se o ganho observado na unidade 0 se generaliza; (2) teste de *fine-tuning* do FlowState, caso venha a ser disponibilizado em versões futuras da biblioteca *granite-tsfm*; (3) incorporação de covariáveis meteorológicas — particularmente radiação solar prevista —, que exigiria modificação no protocolo de avaliação para fornecer valores futuros das covariáveis durante a inferência; (4) expansão do período de teste para capturar eventos sazonais extremos (ondas de frio, picos de demanda); e (5) replicação da comparação em outros *datasets* de energia, como os do ONS (Brasil) e NREL (EUA), para verificar se a superioridade do FlowState se mantém em diferentes contextos climáticos e de mercado.

**6. Reprodução**

O código-fonte completo está disponível no GitHub sob licença Apache-2.0:

  https://github.com/ifsc-sj-projetos-ia/time-series-foundation-models

O repositório está organizado por fases, conforme documentado em `docs/implementation_plan.md`:

```
src/
├── shared/            # eval.py, datasets.py (utilitários transversais)
├── phase1_data/       # preprocess.py (pré-processamento)
├── phase2_baselines/  # run_zero_shot_l2.py, baselines.py
├── phase3_flowstate/  # run_flowstate.py, scale_sweep.py
├── phase4_finetune/   # run_finetune.py, doe_screening.py
└── phase5_report/     # comparison_tables.py, national_aggregation.py
results/               # métricas por fase
findings/              # documentação cronológica dos achados
```

**Dependências** (Python 3.12, GPU com suporte a CUDA recomendada):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Execução dos experimentos** (a partir da raiz do repositório):

```bash
# Pré-processamento dos dados (gera data/processed/unit_*.pt)
python -m src.phase1_data.preprocess

# TTM2 zero-shot (L2)
python -m src.phase2_baselines.run_zero_shot_l2

# FlowState zero-shot (r1.0, contexto 2048)
python -m src.phase3_flowstate.run_flowstate --context 2048

# FlowState zero-shot (r1.0, contexto 512)
python -m src.phase3_flowstate.run_flowstate --context 512

# FlowState zero-shot (r1.1, contexto 4096)
python -m src.phase3_flowstate.run_flowstate --context 4096 --revision r1.1

# Varredura de escala (5 fatores × 5 unidades representativas)
python -m src.phase3_flowstate.scale_sweep

# DOE de fine-tuning do TTM2 (triagem fatorial 2³)
python -m src.phase4_finetune.doe_screening

# Fine-tuning com janelas densas (todas as 69 unidades)
python -m src.phase4_finetune.validate_stride1

# Tabelas comparativas e agregação nacional
python -m src.phase5_report.comparison_tables
python -m src.phase5_report.national_aggregation
```

**Hardware**: todos os experimentos foram executados em uma GPU NVIDIA GeForce GTX 1650 com 4 GB de VRAM e 8 GB de RAM de sistema. O FlowState r1.0 requer aproximadamente 3,5 GB de VRAM em pico (durante a convolução FFT da camada S5); o TTM2 requer menos de 0,5 GB. Ambos os modelos podem ser executados em CPU, embora com tempo de inferência proporcionalmente maior. Nenhuma infraestrutura em nuvem é necessária para reproduzir os resultados deste relatório.
