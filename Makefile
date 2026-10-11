PY ?= python3
BRIEF ?= examples/brief_example.json
PORT ?= 8501

.PHONY: help test test-ui install-ui verify demo run ui log clean

help:
	@echo "make install-ui install the dashboard extra (streamlit + pandas)"
	@echo "make test      run the full test suite"
	@echo "make test-ui   run only the dashboard tests"
	@echo "make ui        launch the web dashboard on port $(PORT)"
	@echo "make verify    run the input decision gate on $(BRIEF) (no tokens spent)"
	@echo "make demo      run the bundled example brief end to end"
	@echo "make run       run $(BRIEF) through the full loop"
	@echo "make log       summarise change_log.jsonl"
	@echo "make clean     remove run artefacts"

install-ui:
	$(PY) -m pip install -e '.[ui]'

test:
	$(PY) -m pytest tests/ -q

test-ui:
	$(PY) -m pytest tests/test_ui.py -q -rs

ui:
	$(PY) -m articlegen ui --port $(PORT)

verify:
	$(PY) -m articlegen verify $(BRIEF)

demo:
	$(PY) -m articlegen demo --json runs/demo_result.json

run:
	$(PY) -m articlegen run $(BRIEF)

log:
	$(PY) -m articlegen log

clean:
	rm -rf runs/ .pytest_cache/
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
