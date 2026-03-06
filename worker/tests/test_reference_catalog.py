from app.pipeline.reference_catalog import get_adavu_reference


def test_tattadavu_match():
    result = get_adavu_reference("tattadavu", "3rd Tattadavu")
    assert result is not None
    assert "Tattadavu" in result
    assert "aramandi" in result.lower()
    assert "technique" in result.lower()


def test_nattadavu_match():
    result = get_adavu_reference("nattadavu", "3rd Nattadavu")
    assert result is not None
    assert "Nattadavu" in result
    assert "arm" in result.lower()


def test_no_match_returns_general():
    result = get_adavu_reference("alarippu", "Alarippu")
    assert result is not None
    assert "General Bharatanatyam" in result


def test_none_inputs():
    assert get_adavu_reference(None, None) is None


def test_item_name_only():
    result = get_adavu_reference(None, "5th tattadavu practice")
    assert result is not None
    assert "5th" in result.lower() or "Tattadavu" in result


def test_specific_variation_content():
    result = get_adavu_reference("tattadavu", "7th Tattadavu")
    assert "Tai Tai Tat Tat" in result
    assert "8th count pause" in result or "technique" in result.lower()


def test_all_eight_tattadavus():
    result = get_adavu_reference("tattadavu", None)
    assert result is not None
    for n in ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"):
        assert n in result, f"{n} tattadavu missing from catalog"


def test_all_eight_nattadavus():
    result = get_adavu_reference("nattadavu", None)
    assert result is not None
    for n in ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"):
        assert n in result, f"{n} nattadavu missing from catalog"


def test_other_adavu_families():
    for family in ("tattimetti", "kuditta_mettu", "mandi_adavu", "sarikkal", "pakka_adavu", "tirmanam", "paraval_adavu"):
        result = get_adavu_reference(family, None)
        assert result is not None, f"{family} not found in catalog"
        assert "technique" in result.lower(), f"{family} missing technique points"
