"""
Hidden grading tests for the LLM comparison task. Copied into the agent's
workspace only AFTER the agent has finished, so the model never sees them.
"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_create_book_returns_201_with_id():
    response = client.post("/books", json={"title": "Hyperion", "author": "Dan Simmons", "year": 1989})
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Hyperion"
    assert isinstance(body.get("id"), int)
    # and it's actually stored
    assert client.get(f"/books/{body['id']}").status_code == 200


def test_year_out_of_range_is_rejected():
    response = client.post("/books", json={"title": "Future", "author": "Someone", "year": 3000})
    assert response.status_code == 422


def test_empty_title_is_rejected():
    response = client.post("/books", json={"title": "", "author": "Someone", "year": 2000})
    assert response.status_code == 422


def test_missing_book_returns_404():
    response = client.get("/books/9999")
    assert response.status_code == 404


def test_existing_endpoints_still_work():
    assert client.get("/books").status_code == 200
    assert client.get("/books/1").json()["title"] == "Dune"
