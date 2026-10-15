# LLM comparison on the same coding task

Task (identical for every model, run in auto mode on a fresh copy of eval/sample_project):

> This folder is a small FastAPI app (app.py) with tests in test_app.py. Make these changes:
> 1. Add a POST /books endpoint that accepts a JSON body validated with a Pydantic model:
>    title (non-empty string), author (string), year (integer between 1450 and 2100).
>    Store the new book with the next free id and return it with status code 201.
> 2. GET /books/{book_id} crashes with a 500 error when the id doesn't exist. Make it return a 404 error instead.
> Check the FastAPI docs for how to do these. Add tests for both changes to test_app.py,
> then run the tests with `python -m pytest -q` and fix anything that fails.

| | groq:openai/gpt-oss-120b | groq:openai/gpt-oss-20b | groq:qwen/qwen3.8-27b |
|---|---|---|---|
| Hidden grading tests passed | 5/5 | 1/5 | 5/5 |
| Used the docs search (docs-rag) | no | no | yes |
| Agent's own tests (pass/fail) | 4/0 | 2/0 | 0/1 |
| Time (s) | 245.1 | 63.0 | 536.0 |
| Model calls | 11 | 7 | 13 |
| Tool calls (failed) | 10 (0) | 3 (0) | 13 (1) |
| Calls per server | filesystem 9, local 1 | filesystem 3 | filesystem 10, docs-rag 3 |
| Tokens in/out | 41643/2465 | n/a | n/a |
| Hit step limit | no | no | no |
| Automatic recoveries | 0 | 3 | 1 |
| Errors that stopped the task | 0 | 1 | 1 |
