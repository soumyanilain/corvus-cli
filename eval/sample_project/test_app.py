from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_list_books():
    response = client.get("/books")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_book():
    response = client.get("/books/1")
    assert response.status_code == 200
    assert response.json()["title"] == "Dune"
