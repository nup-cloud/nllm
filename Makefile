.PHONY: setup test model-download model-verify model-create deploy clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

model-download:
	cd models && bash verify.sh download

model-verify:
	cd models && bash verify.sh check

model-create:
	ollama create nllm -f deploy/Modelfile

deploy:
	cd deploy && docker compose up -d

clean:
	rm -rf build/ dist/ *.egg-info/ src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
