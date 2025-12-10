.PHONY: test clean build sync

test:
	uv run pytest

clean:
	rm -rf build dist *.egg-info __pycache__ .pytest_cache .venv
	rm -f pylibsparseir/*.dylib pylibsparseir/*.so pylibsparseir/*.dll pylibsparseir/*.h

build:
	uv build

sync:
	uv sync
