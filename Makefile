# Variables
PYTHON_VERSION=3.11

.PHONY: help clean install format patch minor major build publish publish-dryrun test

clean: ## Clean up build, test, and coverage artifacts
	@echo "🚀 Cleaning up build, test, and coverage artifacts"
	rm -rf .venv/
	rm -rf dist/
	rm -rf src/*.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

install: clean ## Install dependencies and create virtual environment
	@echo "🚀 Installing dependencies from pyproject.toml"
	@uv sync && uvx pre-commit install

check: ## Check code quality and consistency
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uvx pre-commit run -a
	@echo "🚀 Static type checking: Running pyright"
	@uv run pyright

format: ## Format code using pre-commit hooks
	@uvx pre-commit run --all-files

patch: ## Bump version patch
	@uvx bump-my-version bump patch
	uv lock --upgrade

minor: ## Bump version minor
	@uvx bump-my-version bump minor
	uv lock --upgrade

major: ## Bump version major
	@uvx bump-my-version bump major
	uv lock --upgrade

pre-release: ## Bump version pre-release
	@uvx bump-my-version bump pre_l
	uv lock --upgrade

pre-release-num: ## Bump version pre-release
	@uvx bump-my-version bump pre_n
	uv lock --upgrade

build: install ## Build the package
	@echo "🚀 Building the package"
	@uv build

publish: build ## Publish the package to PyPI
	@uv publish --token $(PYPI_API_TOKEN)

publish-dryrun: build ## Dry run of publishing the package to PyPI
	@uv publish --index testpypi --token $(TEST_PYPI_API_TOKEN)

# TODO: Check if it will run the pytest from the correct venv
test: ## Run tests using pytest
	@echo "🚀 Running tests with pytest"
	@uv run --frozen pytest --doctest-modules --cov=fmuloader --cov-report=term --cov-report=xml

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help

# Add dev packages with `uv add xxx --dev`
# Run executable with `uvx xxx`
