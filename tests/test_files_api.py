from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.app import app
from rag.config import settings
from rag.models import SearchHit

client = TestClient(app)


def test_download_image_ok_and_traversal_blocked(tmp_path, monkeypatch):
    imgdir = tmp_path / "images"
    imgdir.mkdir()
    (imgdir / "a.png").write_bytes(b"\x89PNG\r\n")
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    res = client.get("/files/images/a.png")
    assert res.status_code == 200
    assert res.content.startswith(b"\x89PNG")

    assert client.get("/files/images/../x.pdf").status_code in (404, 422)
    assert client.get("/files/images/nope.png").status_code == 404


def test_download_file_ok(tmp_path, monkeypatch):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF")
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    res = client.get("/files/doc.pdf")
    assert res.status_code == 200
    assert res.content == b"%PDF"


def test_search_rewrites_image_url(monkeypatch):
    svc = MagicMock()
    svc.search.return_value = [
        SearchHit(
            content="cap",
            source="d.pdf",
            score=0.9,
            block_type="image",
            image_path="data/uploads/images/a.png",
        )
    ]
    monkeypatch.setattr(settings, "public_base_url", "http://rag:8000")
    with patch("api.app.get_search_service", return_value=svc):
        res = client.post("/search", json={"query": "图", "top_k": 1})
    assert res.status_code == 200
    assert res.json()["results"][0]["image_path"] == "http://rag:8000/files/images/a.png"


def test_search_keeps_relative_without_base(monkeypatch):
    svc = MagicMock()
    svc.search.return_value = [
        SearchHit(
            content="cap",
            source="d.pdf",
            score=0.9,
            block_type="image",
            image_path="data/uploads/images/a.png",
        )
    ]
    monkeypatch.setattr(settings, "public_base_url", "")
    with patch("api.app.get_search_service", return_value=svc):
        res = client.post("/search", json={"query": "图", "top_k": 1})
    assert res.json()["results"][0]["image_path"] == "data/uploads/images/a.png"
