# BARACHO-RL

BARACHO-RL é um framework minimalista (MVP) para prototipar agentes de RL focados em negócios,
com horizonte longo e métricas financeiras. Esta versão inclui:

- `DynamicPricingEnv` — ambiente simples com sazonalidade e elasticidade-preço
- `Trainer` — laço de treino e relatório básico (CSV)
- `AgentBase`, `RandomAgent`, `RuleAgent` — exemplos
- `Simulator` — stub para usar logs históricos ou gerar dados sintéticos
- CLI `baracho` com comandos `train`, `simulate`, `report`

> Objetivo: **3 linhas para treinar**, **≤15 linhas para criar um agente customizado**.

## Instalação de desenvolvimento
```bash
pip install -e .
```

## Exemplo rápido
```bash
python examples/pricing_quickstart.py
```


## Treino declarativo (Compose + YAML)
```bash
baracho train --config configs/compose_pricing_hiring_cash.yaml --steps 20000 --outdir runs/compose_yaml
baracho refine --config configs/compose_pricing_hiring_cash.yaml --episodes 40 --window 3 --outdir runs/refine_yaml
```
