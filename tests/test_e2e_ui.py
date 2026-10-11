"""End-to-end: API FastAPI sungguhan (uvicorn) + antarmuka Streamlit (AppTest) + database SQLite."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_api(tmp_path_factory):
    port = _free_port()
    env = {**os.environ, "DB_PATH": str(tmp_path_factory.mktemp("db") / "e2e.db")}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(80):
        try:
            if requests.get(url + "/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("API tidak berhasil start")
    yield url
    proc.terminate()
    proc.wait(timeout=10)


def _click(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def test_full_flow_ui_api_database(live_api, monkeypatch):
    monkeypatch.setenv("API_URL", live_api)
    at = AppTest.from_file(str(ROOT / "streamlit_app" / "app.py"), default_timeout=60).run()
    assert not at.exception

    # 1) pilih gejala -> prediksi -> hasil tampil
    at.multiselect(key="symptoms").set_value(["back pain", "low back pain", "leg pain", "hip pain"])
    at = _click(at, "Prediksi")
    assert not at.exception and not at.error
    assert "Diagnosis" in at.dataframe[0].value.columns

    # 2) prediksi tersimpan di database dan muncul di halaman riwayat
    assert requests.get(live_api + "/stats", timeout=5).json()["total_predictions"] == 1
    at.sidebar.radio[0].set_value("Riwayat").run()
    assert not at.exception
    assert at.metric[0].value == "1"

    # 3) halaman performa model dapat dirender
    at.sidebar.radio[0].set_value("Performa Model").run()
    assert not at.exception
