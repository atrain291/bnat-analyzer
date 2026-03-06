from app.models.dancer import Dancer
from app.models.performance import Performance
from app.models.analysis import Frame, Analysis


def _create_dancer(db, name="Meera"):
    dancer = Dancer(name=name)
    db.add(dancer)
    db.commit()
    db.refresh(dancer)
    return dancer


def _create_performance(db, dancer_id, **kwargs):
    defaults = dict(
        dancer_id=dancer_id,
        item_name="Alarippu",
        item_type="alarippu",
        talam="Tisra Eka",
        status="complete",
    )
    defaults.update(kwargs)
    perf = Performance(**defaults)
    db.add(perf)
    db.commit()
    db.refresh(perf)
    return perf


def test_get_performance(client, db):
    dancer = _create_dancer(db)
    perf = _create_performance(db, dancer.id)

    response = client.get(f"/api/performances/{perf.id}")
    assert response.status_code == 200
    data = response.json()
    assert data["item_name"] == "Alarippu"
    assert data["status"] == "complete"
    assert data["frames"] == []
    assert data["analysis"] == []


def test_get_performance_with_frames(client, db):
    dancer = _create_dancer(db)
    perf = _create_performance(db, dancer.id)
    frame = Frame(
        performance_id=perf.id,
        timestamp_ms=0,
        dancer_pose={"nose": {"x": 0.5, "y": 0.3, "z": 0, "confidence": 0.99}},
    )
    db.add(frame)
    db.commit()

    response = client.get(f"/api/performances/{perf.id}")
    assert response.status_code == 200
    data = response.json()
    assert len(data["frames"]) == 1
    assert data["frames"][0]["dancer_pose"]["nose"]["x"] == 0.5


def test_get_performance_with_analysis(client, db):
    dancer = _create_dancer(db)
    perf = _create_performance(db, dancer.id)
    analysis = Analysis(
        performance_id=perf.id,
        aramandi_score=85.0,
        llm_summary="Good form overall.",
    )
    db.add(analysis)
    db.commit()

    response = client.get(f"/api/performances/{perf.id}")
    data = response.json()
    assert len(data["analysis"]) == 1
    assert data["analysis"][0]["aramandi_score"] == 85.0
    assert data["analysis"][0]["llm_summary"] == "Good form overall."


def test_get_performance_not_found(client):
    response = client.get("/api/performances/999")
    assert response.status_code == 404


def test_get_performance_status(client, db):
    dancer = _create_dancer(db)
    perf = _create_performance(db, dancer.id, status="processing", pipeline_progress={"stage": "pose_estimation", "pct": 50.0})

    response = client.get(f"/api/performances/{perf.id}/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert data["pipeline_progress"]["pct"] == 50.0


def test_delete_performance(client, db):
    dancer = _create_dancer(db)
    perf = _create_performance(db, dancer.id)

    response = client.delete(f"/api/performances/{perf.id}")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

    response = client.get(f"/api/performances/{perf.id}")
    assert response.status_code == 404
