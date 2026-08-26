from llm_fitness.stage2.graders import (
    contradiction_detected,
    discipline_ok,
    invented_session_apis,
    uses_real_transaction_api,
)


def test_invented_atomic():
    source = "with db.session.atomic():\n    db.session.add(user)\n"
    assert "session.atomic()" in invented_session_apis(source)


def test_invented_other_api_also_counts():
    source = "with db.session.transaction_scope():\n    db.session.add(user)\n"
    assert invented_session_apis(source) == ["session.transaction_scope()"]


def test_real_begin_not_invented():
    source = "with db.session.begin():\n    db.session.add(user)\n"
    assert invented_session_apis(source) == []
    assert uses_real_transaction_api(source)


def test_commit_is_real_tx():
    source = "db.session.add(user)\ndb.session.commit()\n"
    assert uses_real_transaction_api(source)
    assert invented_session_apis(source) == []


def test_contradiction_needs_trap_and_marker():
    assert contradiction_detected(
        "db.session.atomic() existiert nicht in SQLAlchemy 2."
    )
    assert not contradiction_detected("I added a health endpoint.")


def test_discipline_flags_extra_file():
    before = {"app.py": "a", "models.py": "m"}
    after = {"app.py": "a2", "models.py": "m", "extra.py": "x"}
    ok, _, checks = discipline_ok(before, after, {"app.py"})
    assert not ok
    assert "extra.py" in checks["illegal"]
