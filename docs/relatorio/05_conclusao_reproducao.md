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
