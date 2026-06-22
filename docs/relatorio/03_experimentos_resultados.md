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
