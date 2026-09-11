PY := .venv/bin/python
UV := uv
CONFIG := configs/config.yaml
LOGS := reports/logs

.PHONY: install lint format test data train train-smoke eval eval-detector \
        eval-generalization eval-asr eval-asr-report eval-utility demo all clean

install:
	$(UV) venv --python 3.11
	$(UV) sync --all-groups
	$(PY) -c "import torch; print('cuda:', torch.cuda.is_available())"

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:
	$(UV) run pytest -q

data:
	$(PY) scripts/prepare_data.py --config $(CONFIG)
	$(PY) scripts/build_demo_corpus.py --config $(CONFIG)

# Prova rapida: 250 passi su 4000 esempi. Se eval_loss non scende ben sotto 0.69
# il modello sta collassando e il training lungo e' inutile.
train-smoke:
	$(PY) scripts/train.py --config $(CONFIG) --smoke

# Il log finisce anche su file, cosi' l'avanzamento si segue con:
#   tail -f reports/logs/train.log
train:
	@mkdir -p $(LOGS)
	$(PY) scripts/train.py --config $(CONFIG) 2>&1 | tee $(LOGS)/train.log

eval-detector:
	@mkdir -p $(LOGS)
	$(PY) scripts/evaluate.py --config $(CONFIG) --experiment detector 2>&1 | tee $(LOGS)/detector.log

eval-generalization:
	@mkdir -p $(LOGS)
	$(PY) scripts/evaluate.py --config $(CONFIG) --experiment generalization 2>&1 | tee $(LOGS)/generalization.log

# Un run per ogni modello in evaluation.victims (config.yaml); per un solo modello:
#   make eval-asr VICTIM=mistralai/ministral-3-3b
VICTIM_FLAG := $(if $(VICTIM),--victim $(VICTIM),)
eval-asr:
	@mkdir -p $(LOGS)
	$(PY) scripts/evaluate.py --config $(CONFIG) --experiment asr $(VICTIM_FLAG) 2>&1 | tee -a $(LOGS)/asr.log

# Ricostruisce asr.csv, asr_by_position.csv e asr_by_vector.csv dai grezzi gia' presenti.
eval-asr-report:
	$(PY) scripts/evaluate.py --config $(CONFIG) --experiment asr-report

eval-utility:
	@mkdir -p $(LOGS)
	$(PY) scripts/evaluate.py --config $(CONFIG) --experiment utility 2>&1 | tee $(LOGS)/utility.log

eval: eval-detector eval-generalization eval-asr eval-utility

# fileWatcherType=none: il watcher di Streamlit cammina sui moduli lazy di transformers
# e tenta di importare zoedepth, che richiede torchvision (non installato e non necessario).
# Sono traceback innocui ma rumorosi, e rallentano l'avvio.
demo:
	$(UV) run streamlit run app/demo.py --server.fileWatcherType none

all: data train eval

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
