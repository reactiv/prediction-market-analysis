.PHONY: analyze run index transform package lint format setup

RUN = uv run main.py

analyze:
	$(RUN) analyze

run:
	$(RUN) analyze $(filter-out $@,$(MAKECMDGOALS))

index:
	$(RUN) index

transform:
	$(RUN) transform

transform-all:
	$(RUN) transform all

package:
	$(RUN) package

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

setup:
	sh scripts/install-tools.sh
	sh scripts/download.sh

%:
	@:
