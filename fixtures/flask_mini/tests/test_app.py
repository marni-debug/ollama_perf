def test_list_users_empty(client):
    response = client.get("/users")
    assert response.status_code == 200
    assert response.get_json() == []


def test_create_user(client):
    response = client.post("/users", json={"name": "Ada"})
    assert response.status_code == 201
    data = response.get_json()
    assert data["name"] == "Ada"
    assert data["id"]

    listed = client.get("/users")
    assert listed.status_code == 200
    names = [row["name"] for row in listed.get_json()]
    assert "Ada" in names
