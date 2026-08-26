from flask import Flask, jsonify, request

from models import User, db


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    if config:
        app.config.update(config)
    db.init_app(app)

    @app.get("/users")
    def list_users():
        users = db.session.execute(db.select(User)).scalars().all()
        return jsonify([{"id": u.id, "name": u.name} for u in users])

    @app.post("/users")
    def create_user():
        payload = request.get_json(silent=True) or {}
        name = payload.get("name")
        if not name:
            return jsonify({"error": "name required"}), 400
        user = User(name=name)
        db.session.commit()
        return jsonify({"id": user.id, "name": user.name}), 201

    with app.app_context():
        db.create_all()

    return app


if __name__ == "__main__":
    create_app().run()
