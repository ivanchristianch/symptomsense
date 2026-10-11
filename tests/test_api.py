import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["n_classes"] == 254


@pytest.mark.parametrize("model", ["dnn", "logreg"])
def test_predict_returns_sorted_top_k(client, model):
    r = client.post("/predict", json={"symptoms": ["back pain", "leg pain", "hip pain"], "model": model, "top_k": 3})
    assert r.status_code == 200
    probs = [t["probability"] for t in r.json()["top_predictions"]]
    assert len(probs) == 3
    assert probs == sorted(probs, reverse=True)
    assert r.json()["disclaimer"]


def test_unknown_symptom_is_422(client):
    assert client.post("/predict", json={"symptoms": ["bukan gejala"]}).status_code == 422


def test_empty_symptoms_is_422(client):
    assert client.post("/predict", json={"symptoms": []}).status_code == 422


def test_prediction_is_stored_and_listed(client):
    for _ in range(2):
        assert client.post("/predict", json={"symptoms": ["cough", "fever"]}).status_code == 200
    assert len(client.get("/history").json()["items"]) == 2
    assert client.get("/stats").json()["total_predictions"] == 2


def test_save_false_does_not_store(client):
    client.post("/predict", json={"symptoms": ["cough"], "save": False})
    assert client.get("/stats").json()["total_predictions"] == 0


def test_models_endpoint_lists_comparison(client):
    assert len(client.get("/models").json()["comparison"]) == 4
