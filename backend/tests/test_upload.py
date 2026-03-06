import io
from unittest.mock import patch

from app.models.dancer import Dancer


def _create_dancer(db, name="Meera"):
    dancer = Dancer(name=name)
    db.add(dancer)
    db.commit()
    db.refresh(dancer)
    return dancer


@patch("app.api.routes.upload.dispatch_pipeline", return_value="fake-task-id")
def test_upload_video(mock_dispatch, client, db, tmp_path):
    dancer = _create_dancer(db)
    video_content = b"fake video content"

    response = client.post(
        "/api/upload/",
        data={
            "dancer_id": str(dancer.id),
            "item_name": "Alarippu",
            "item_type": "alarippu",
            "talam": "Tisra Eka",
        },
        files={"file": ("test.mp4", io.BytesIO(video_content), "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "fake-task-id"
    assert data["status"] == "queued"
    assert "performance_id" in data
    mock_dispatch.assert_called_once()


@patch("app.api.routes.upload.dispatch_pipeline", return_value="fake-task-id")
def test_upload_invalid_extension(mock_dispatch, client, db):
    dancer = _create_dancer(db)

    response = client.post(
        "/api/upload/",
        data={"dancer_id": str(dancer.id)},
        files={"file": ("test.txt", io.BytesIO(b"not a video"), "text/plain")},
    )
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


@patch("app.api.routes.upload.dispatch_pipeline", return_value="fake-task-id")
def test_upload_dancer_not_found(mock_dispatch, client, db):
    response = client.post(
        "/api/upload/",
        data={"dancer_id": "999"},
        files={"file": ("test.mp4", io.BytesIO(b"video"), "video/mp4")},
    )
    assert response.status_code == 404
