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
