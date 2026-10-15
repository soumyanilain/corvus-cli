"""Bookshelf API: a tiny FastAPI app used for Corvus demos and the LLM comparison."""

from fastapi import FastAPI

app = FastAPI(title="Bookshelf API")

BOOKS: dict[int, dict] = {
    1: {"id": 1, "title": "Dune", "author": "Frank Herbert", "year": 1965},
    2: {"id": 2, "title": "Neuromancer", "author": "William Gibson", "year": 1984},
}


@app.get("/books")
def list_books():
    """Return every book on the shelf."""
    return list(BOOKS.values())


@app.get("/books/{book_id}")
def get_book(book_id: int):
    """Return one book by id."""
    return BOOKS[book_id]
