# Plano de Refatoração — Fase 6

## Motivação

A fase 1 de pré-processamento dividiu cada unidade de previsão em arquivos `.pt` e `.csv` individuais. Isso trouxe duas limitações:

1. **Impossibilidade de usar `id_columns` do TTM2**: o modelo aceita um identificador de série temporal que permite treinar/inferir sobre múltiplas séries em um único *pipeline*. Ao fragmentar os dados por unidade, perdemos essa capacidade.

2. **Dificuldade de auditoria**: 138 arquivos (69 `.pt` + 69 `.csv`) para gerenciar, em vez de dois CSVs planos.

A fase 6 refatora o pré-processamento e os experimentos para usar um formato de painel longo (*long format panel*), alinhado com a API do `TimeSeriesPreprocessor` do `granite-tsfm`.

## Mudanças

### 1. Pré-processamento (`preprocess_v2.py`)

**Entrada**: CSVs brutos do Kaggle (`train.csv`, `forecast_weather.csv`, `historical_weather.csv`, `electricity_prices.csv`, `client.csv`)

**Saída**:
- `data/processed/generation.csv` — registros com `is_consumption=0`
- `data/processed/consumption.csv` — registros com `is_consumption=1`

**Pipeline**:
1. Carregar `train.csv` completo
2. Filtrar por `is_consumption` → geração vs consumo
3. Juntar `client.csv` (capacidade instalada, número de instalações)
4. Juntar `historical_weather` (média por condado por Haversine)
5. Juntar `forecast_weather` (previsão de menor *hours_ahead*, média por condado)
6. Juntar `electricity_prices` (preço horário)
7. Dividir 70/15/15 (treino/validação/teste) por tempo (não aleatório)
8. Exportar CSV único

**Colunas em cada CSV**:
```
prediction_unit_id, datetime, target, county, is_business, product_type,
temperature, dewpoint, rain, cloudcover_total, windspeed_10m, shortwave_radiation,
temperature_forecast, cloudcover_total_forecast,
price, eic_count, installed_capacity,
split
```

### 2. Configuração (`config.py`)

Colunas, hiperparâmetros e divisão centralizados. Importado pelos scripts de experimento.

### 3. Experimentos (a fazer depois)

- `run_ttm2_v2.py`: TTM2 zero-shot com `id_columns=["prediction_unit_id"]`, via `TimeSeriesPreprocessor` + `TimeSeriesForecastingPipeline`
- `run_flowstate_v2.py`: FlowState zero-shot iterando por `prediction_unit_id` (loop), lendo do CSV

### 4. Resultados

`results/phase6/ttm2_zero/metrics.csv` etc. Para comparação com `results/phase2/l2_zero_shot/`.

## O que NÃO muda

- Nada em `src/phase1_data/` a `src/phase5_report/`
- Nada em `results/phase1/` a `results/phase5/`
- Os arquivos `data/processed/unit_*.pt` e `data/processed/unit_*.csv` existentes (auditoria)

## Estrutura da Fase 6

```
src/phase6_refactor/
├── __init__.py
├── config.py              # Colunas, paths, split ratio
├── preprocess_v2.py       # Novo pré-processamento
├── run_ttm2_v2.py         # TTM2 com id_columns (futuro)
└── run_flowstate_v2.py    # FlowState do CSV (futuro)
```
