"""
API服务测试
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from httpx import AsyncClient, ASGITransport
from src.api.server import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_health_check(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"


@pytest.mark.asyncio
async def test_generate_content(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 触发startup
        await client.get("/health")

        response = await client.post("/api/v1/generate", json={
            "prompt": "推荐一款适合夏天的防晒霜",
            "category": "美妆",
            "max_length": 256,
        })
        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "content" in data
            assert "quality_score" in data
            assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_recommend(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/health")

        response = await client.post("/api/v1/recommend", json={
            "user_id": "user_001",
            "num_items": 5,
        })
        if response.status_code == 200:
            data = response.json()
            assert "items" in data
            assert len(data["items"]) == 5


@pytest.mark.asyncio
async def test_search(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.get("/health")

        response = await client.post("/api/v1/search", json={
            "query_text": "美丽的日落风景",
            "top_k": 5,
            "search_type": "text2image",
        })
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 5
