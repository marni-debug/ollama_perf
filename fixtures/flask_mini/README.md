# Mini user service

Small Flask + SQLAlchemy app.

## Endpoints

- `GET /users` — list users
- `POST /users` — JSON `{"name": "..."}`, returns 201

## Transactions

Persist writes with the SQLAlchemy session (`add` / `commit`). Run tests with `pytest`.
