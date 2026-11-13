.PHONY: check standards sync-standards
SHELL := bash
ORG = https://github.com/lowmagic/.github

check:
	uv lock --check
	uv run ruff check
	uv run ruff format --check
	uv run mypy
	uv run pytest -rs
	git ls-files -z '*.sh' | xargs -0 -r shellcheck

standards:
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	git clone -q --depth 1 --branch main $(ORG) "$$tmp" && \
	"$$tmp/code-standards/sync.sh" check "$$tmp"

sync-standards:
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	git clone -q --depth 1 --branch main $(ORG) "$$tmp" && \
	"$$tmp/code-standards/sync.sh" sync "$$tmp" "$$(git -C "$$tmp" rev-parse HEAD)"
