# Project Instructions

- This project uses the repository-local virtual environment at `.venv/`.
- For Python commands, prefer `.venv/bin/python` instead of bare `python` or `python3`.
- For pip commands, prefer `.venv/bin/python -m pip`.
- For a full local install, use `.venv/bin/python -m pip install -r requirements-cu128.txt`.
- If a command needs an activated environment, run `source .venv/bin/activate` first and then use normal `python` / `pip`.
- When verifying Python changes, use the virtualenv interpreter so imports and installed packages match the project environment.
- Run Python tests, lint, type checks, and syntax checks through `.venv/bin/python` so verification uses the project environment consistently.
