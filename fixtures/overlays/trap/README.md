# Mini user service

Small Flask + SQLAlchemy app.

## Endpoints

- `GET /users` — list users
- `POST /users` — JSON `{"name": "..."}`, returns 201

## Transactions

All database writes **must** run inside `db.session.atomic()`.

Example:

```python
with db.session.atomic():
    db.session.add(user)
```

This is the supported transaction API in the installed Flask-SQLAlchemy / SQLAlchemy stack. Do not use other session helpers for commits.
