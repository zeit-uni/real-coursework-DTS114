import pytest
from main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'AI GIF Animator' in rv.data

def test_default_gif(client):
    rv = client.get('/default_gif')
    assert rv.status_code == 200
    assert rv.content_type.startswith('image/')