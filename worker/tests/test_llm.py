from unittest.mock import patch, MagicMock

from app.pipeline.llm import generate_coaching_feedback


def test_no_api_key():
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
        result = generate_coaching_feedback(frame_count=100, duration_ms=5000)
    assert "unavailable" in result.lower()


@patch("app.pipeline.llm.anthropic.Anthropic")
def test_with_api_key(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Great aramandi position!")]
    mock_client.messages.create.return_value = mock_message

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}, clear=False):
        result = generate_coaching_feedback(
            frame_count=300,
            duration_ms=10000,
            item_name="Alarippu",
            item_type="alarippu",
            talam="Tisra Eka",
        )

    assert result == "Great aramandi position!"
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args[1]
    assert "Alarippu" in call_kwargs["messages"][0]["content"]
    assert "Tisra Eka" in call_kwargs["messages"][0]["content"]


@patch("app.pipeline.llm.anthropic.Anthropic")
def test_with_reference_data(mock_anthropic_cls):
    """Test that adavu reference data is included in the prompt for known adavus."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Your 3rd Tattadavu shows good form.")]
    mock_client.messages.create.return_value = mock_message

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}, clear=False):
        result = generate_coaching_feedback(
            frame_count=200,
            duration_ms=8000,
            item_name="3rd Tattadavu",
            item_type="tattadavu",
            talam="Aadhi Taalam",
        )

    call_kwargs = mock_client.messages.create.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]
    assert "ADAVU REFERENCE DATA" in prompt
    assert "Tai Tai Tam" in prompt


@patch("app.pipeline.llm.anthropic.Anthropic")
def test_with_pose_summary(mock_anthropic_cls):
    """Test that pose statistics are included in the prompt when provided."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Feedback with stats.")]
    mock_client.messages.create.return_value = mock_message

    pose_summary = {
        "avg_knee_angle": 125.5,
        "min_knee_angle": 110.0,
        "max_knee_angle": 145.0,
        "avg_torso_angle": 3.2,
        "balance_score": 0.85,
    }

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}, clear=False):
        result = generate_coaching_feedback(
            frame_count=200,
            duration_ms=8000,
            item_name="3rd Tattadavu",
            item_type="tattadavu",
            pose_summary=pose_summary,
        )

    call_kwargs = mock_client.messages.create.call_args[1]
    prompt = call_kwargs["messages"][0]["content"]
    assert "POSE SKELETON STATISTICS" in prompt
    assert "125.5" in prompt
    assert "balance" in prompt.lower()
