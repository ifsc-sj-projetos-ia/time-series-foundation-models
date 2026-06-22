# time-series-foundation-models

Comparação entre dois modelos de fundação para séries temporais da família IBM Granite — **TinyTimeMixer v2 (TTM2)** e **FlowState** — na previsão horária de produção e consumo de energia elétrica de prosumers na Estônia, utilizando o dataset Enefit (*Predict Energy Behavior of Prosumers*, Kaggle).

Licença: Apache-2.0.

---

## Objetivo

Avaliar e comparar o desempenho de dois *time series foundation models* em modo *zero-shot* e *fine-tuning* sobre dados reais de energia, quantificando o ganho da arquitetura *State Space Model* (FlowState) sobre *MLP Mixer* (TTM2) e identificando os fatores que mais influenciam a acurácia das previsões.

---

## Métricas Principais

| Modelo | Alvo | MAE | RMSE | SMAPE |
|---|---|---|---|---|
| Persistence | Export | 420,6 | 634,8 | 159,5% |
| Persistence | Import | 136,5 | 172,3 | 60,5% |
| Seasonal Naive | Export | 238,3 | 373,0 | 94,0% |
| Seasonal Naive | Import | 102,5 | 145,1 | 39,9% |
| TTM2 *zero-shot* | Export | 202,3 | 302,9 | 106,4% |
| TTM2 *zero-shot* | Import | 107,7 | 141,6 | 46,2% |
| **FlowState r1.0 (ctx2048)** | **Export** | **181,7** | **282,8** | **99,6%** |
| **FlowState r1.0 (ctx2048)** | **Import** | **92,4** | **128,8** | **39,6%** |

> Média sobre as 69 unidades de previsão. *Export* = geração (injeção na rede), *Import* = consumo (retirada da rede).

O FlowState r1.0 com contexto de 2048 *timesteps* é o melhor modelo do estudo. Em consumo, o *Seasonal Naive* (repetição do ciclo de 24 horas) é competitivo (SMAPE 39,9%), confirmando que essa variável tem forte componente periódica diária. O relatório completo está em `docs/relatorio/relatorio_completo.md`.

---

## Estrutura do Repositório

```
src/
├── shared/               # eval.py, datasets.py (utilitários transversais)
├── phase1_data/          # preprocess.py (pré-processamento)
├── phase2_baselines/     # baselines.py, run_zero_shot_l2.py
├── phase3_flowstate/     # run_flowstate.py, scale_sweep.py
├── phase4_finetune/      # run_finetune.py, doe_screening.py, validate_stride1.py
└── phase5_report/        # comparison_tables.py, figures.py, national_aggregation.py
docs/
├── project_plan.md       # Planejamento conceitual
├── implementation_plan.md # Plano de execução por fases
├── doe_best_practices.md  # Metodologia de Design of Experiments
└── relatorio/             # Relatório final (seções em Markdown)
findings/                  # Documentação cronológica dos achados (01–09)
results/
├── phase2/l2_zero_shot/   # Métricas TTM2 zero-shot
├── phase3/                # Métricas FlowState + varredura de escala
├── phase4/doe_screening/  # Resultados do DOE de fine-tuning
├── phase4/fullshot/       # Resultados full-shot
├── phase5/comparison/     # Tabelas comparativas
├── phase5/national/       # Agregação nacional
└── figures/               # Figuras para o relatório
```

---

## Dependências

Python 3.12. GPU com suporte a CUDA recomendada (GTX 1650 ou superior).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pacotes principais: `granite-tsfm>=0.3`, `torch>=2.0`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `geopandas`.

---

## Instruções de Execução

### 1. Pré-processamento dos dados

O dataset da Enefit deve estar na máquina local em `predict-energy-behavior-of-prosumers/`. O script de pré-processamento gera os arquivos `data/processed/unit_*.pt` e `data/processed/unit_*.csv` (um par por unidade de previsão).

```bash
python -m src.phase1_data.preprocess
```

### 2. Experimentos zero-shot

```bash
# TTM2 zero-shot (revisão L2, 512-96-ft-r2.1)
python -m src.phase2_baselines.run_zero_shot_l2

# FlowState zero-shot (r1.0, contexto 2048 — melhor configuração)
python -m src.phase3_flowstate.run_flowstate --context 2048

# FlowState zero-shot (r1.0, contexto 512 — comparação justa com TTM2)
python -m src.phase3_flowstate.run_flowstate --context 512

# FlowState zero-shot (r1.1, contexto 4096)
python -m src.phase3_flowstate.run_flowstate --context 4096 --revision r1.1

# Varredura de escala (5 fatores × 5 unidades representativas)
python -m src.phase3_flowstate.scale_sweep
```

### 3. Fine-tuning do TTM2

```bash
# DOE de triagem (fatorial 2³, 10 execuções na unidade 0)
python -m src.phase4_finetune.doe_screening

# Validação com janelas densas (stride=1, todas as 69 unidades)
python -m src.phase4_finetune.validate_stride1
```

### 4. Geração de tabelas, figuras e agregação nacional

```bash
# Tabelas comparativas
python -m src.phase5_report.comparison_tables

# Agregação nacional (soma das 69 unidades)
python -m src.phase5_report.national_aggregation

# Figuras para o relatório
python -m src.phase5_report.figures
```

### 5. Hardware

Todos os experimentos foram executados em uma GPU NVIDIA GeForce GTX 1650 (4 GB VRAM, 8 GB RAM). O FlowState r1.0 requer aproximadamente 3,5 GB de VRAM em pico durante a convolução FFT da camada S5; o TTM2 requer menos de 0,5 GB. Ambos os modelos podem ser executados em CPU, com tempo de inferência proporcionalmente maior. Nenhuma infraestrutura em nuvem é necessária.

---

## Limitações

- **Escopo geográfico**: o dataset cobre exclusivamente a Estônia. Os resultados não são generalizáveis para outros países ou climas sem validação independente.
- **Período de teste curto**: 96 horas (4 dias) não capturam eventos sazonais extremos.
- **Covariáveis não utilizadas**: variáveis meteorológicas não foram incorporadas — o FlowState não suporta canais exógenos em modo *zero-shot*, e o módulo *Forecast Channel Mixing* do TTM2 requer valores futuros das covariáveis durante a inferência, indisponíveis no período de teste.
- **Validação parcial do fine-tuning**: o TTM2 *fine-tuned* com janelas densas foi validado apenas na unidade 0 devido ao custo computacional (~5 horas para as 69 unidades).
- **Zero-shot apenas**: o FlowState em sua versão atual não oferece suporte a *fine-tuning*.

---

## Trabalhos Futuros

1. Concluir a validação do TTM2 *fine-tuned* com janelas densas nas 69 unidades de previsão.
2. Testar *fine-tuning* do FlowState quando disponível em versões futuras da biblioteca *granite-tsfm*.
3. Incorporar covariáveis meteorológicas (radiação solar prevista) nos experimentos.
4. Expandir o período de teste para capturar eventos sazonais extremos.
5. Replicar a comparação em outros datasets de energia (ONS Brasil, NREL EUA).
6. Testar a revisão L1 do TTM2 (*512-96-ft-l1-r2.1*) nos experimentos *zero-shot* — os benchmarks do projeto LoadPrediction indicam ganho consistente de 5–10% relativos de MAPE com L1.
