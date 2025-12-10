.PHONY: test clean build

test:
	rm -rf src && python3 prepare_build.py && uv run pytest

clean:
	rm -rf src build dist *.egg-info __pycache__ .pytest_cache

build:
	rm -rf src && python3 prepare_build.py && uv build

sync:
	rm -rf src && python3 prepare_build.py && uv sync
