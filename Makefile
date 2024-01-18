clean:
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache && rm -f .coverage coverage.xml && find . -type d -name __pycache__ -exec rm -r {} +

isort-format:
	isort .

isort-check:
	isort --check-only .

black-format:
	black .

black-check:
	black . --check 

ruff-format:
	ruff --fix .

ruff-check:
	ruff .

mypy:
	mypy .

format: isort-format black-format ruff-format

lint: isort-check black-check ruff-check mypy

test:
	pytest -ra --cov=src --cov-report=term --cov-report=xml

.PHONY: clean format isort test
