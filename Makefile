.PHONY: help chat test
help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

chat:  ## talk to the agent
	python -m app.cli

test:  ## unit tests for the harness
	python tests/test_agent.py
