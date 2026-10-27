.PHONY: all experiment figures test clean

all: experiment figures

experiment:
	python experiments/run_experiment.py --threshold 0.55 --top-k 4 --out results

figures:
	python scripts/make_figures.py --results results --out results/figures --paper-out paper/figures

test:
	pytest -q

clean:
	rm -rf results/*.csv results/*.json results/figures/*.png paper/figures/*.png .pytest_cache ragguard/__pycache__ experiments/__pycache__ scripts/__pycache__ tests/__pycache__
