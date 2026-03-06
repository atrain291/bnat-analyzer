def test_create_dancer(client):
    response = client.post("/api/dancers/", json={
        "name": "Meera",
        "experience_level": "intermediate",
    })
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Meera"
    assert data["experience_level"] == "intermediate"
    assert "id" in data
    assert "created_at" in data


def test_create_dancer_minimal(client):
    response = client.post("/api/dancers/", json={"name": "Priya"})
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Priya"
    assert data["experience_level"] is None


def test_list_dancers_empty(client):
    response = client.get("/api/dancers/")
    assert response.status_code == 200
    assert response.json() == []


def test_list_dancers(client):
    client.post("/api/dancers/", json={"name": "Meera"})
    client.post("/api/dancers/", json={"name": "Priya"})
    response = client.get("/api/dancers/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    names = {d["name"] for d in data}
    assert names == {"Meera", "Priya"}


def test_get_dancer(client):
    create = client.post("/api/dancers/", json={"name": "Meera"})
    dancer_id = create.json()["id"]
    response = client.get(f"/api/dancers/{dancer_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Meera"


def test_get_dancer_not_found(client):
    response = client.get("/api/dancers/999")
    assert response.status_code == 404
