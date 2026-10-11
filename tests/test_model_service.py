import numpy as np
import pytest

from app.model_service import UnknownSymptomError


@pytest.mark.parametrize("model", ["dnn", "logreg"])
def test_matches_original_models(service, golden, model):
    """Implementasi NumPy harus sama dengan model asli (Keras / scikit-learn) pada kasus golden."""
    for case in golden:
        top = service.predict(case["symptoms"], model=model, top_k=1)["top_predictions"][0]
        assert top["disease"] == case[model]["top1"]
        assert top["probability"] == pytest.approx(case[model]["p"], abs=1e-3)


@pytest.mark.parametrize("model", ["dnn", "logreg"])
def test_probabilities_sum_to_one(service, model):
    x = np.zeros((2, len(service.symptoms)), dtype=np.float32)
    x[0, :3] = 1
    p = service.proba(model, x)
    assert p.shape == (2, len(service.classes))
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_unknown_symptom_raises(service):
    with pytest.raises(UnknownSymptomError):
        service.predict(["bukan gejala"], model="dnn")


def test_explanation_covers_every_active_symptom(service):
    syms = service.symptoms[:4]
    res = service.predict(syms, model="logreg")
    assert {e["symptom"] for e in res["explanation"]} == set(syms)
