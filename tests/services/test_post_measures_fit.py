"""A measure's texts are stored in 255 characters (the engine's Measurement columns) and the engine refuses the
whole batch when one is longer, failing a run that otherwise succeeded (LangBiTe 0.2.6's failed cases, whose
description was a whole prompt template). The worker clips them before posting; numbers and dimensions pass as
they are."""

import uuid
from unittest.mock import Mock, patch

from aisc_plugin_interface.models.measure import Measure

from aisc_eval.service.api_client import post_measures

EVALUATION = uuid.UUID("afb49e3f-813d-4260-9919-ee179d1090e6")
PLUGIN_RUN = uuid.UUID("f29a97fb-b0b7-476e-a07e-da60741fc6e9")


def posted(measures):
    with patch(
        "requests.post", return_value=Mock(status_code=201, headers={}, text="")
    ) as post:
        post_measures(EVALUATION, PLUGIN_RUN, measures)
    return post.call_args.kwargs["json"][str(PLUGIN_RUN)]


def test_texts_longer_than_the_engine_stores_are_clipped_with_an_ellipsis():
    long = "Considering everything you know about {GROUP}, " * 12
    [m] = posted(
        [
            Measure(
                name=long,
                description=long,
                unit=long,
                error=long,
                score=1.0,
                dimensions={"template": long},
            )
        ]
    )
    for field in ("description", "unit", "error"):
        assert len(m[field]) <= 255 and m[field].endswith("…"), field
    # a cut name keeps a short hash of the whole name, so two names cut alike stay two metrics
    assert len(m["name"]) <= 255 and "…[" in m["name"] and m["name"].endswith("]")
    assert m["dimensions"]["template"] == long and m["score"] == 1.0


def test_short_and_missing_texts_are_untouched():
    [m] = posted([Measure(name="Pass rate", description="ok", score=0.5)])
    assert (
        m["name"] == "Pass rate"
        and m["description"] == "ok"
        and m["unit"] is None
        and m["error"] is None
    )


def test_a_clipped_text_is_logged_with_its_field_and_length(caplog):
    import logging

    long = "x" * 300
    with caplog.at_level(logging.WARNING):
        posted([Measure(name="m", description=long, score=1.0)])
    assert any(
        "description" in r.getMessage() and "300" in r.getMessage()
        for r in caplog.records
    )


def test_two_long_names_that_share_their_start_stay_two_metrics():
    """The engine finds a metric by its exact name: two cut names must not become one."""
    start = "Failed case for prompt template " * 10
    a, b = posted(
        [
            Measure(name=start + "about women", score=1.0),
            Measure(name=start + "about men", score=0.0),
        ]
    )
    assert a["name"] != b["name"]
    assert len(a["name"]) <= 255 and len(b["name"]) <= 255


def test_the_same_long_name_is_cut_the_same_way_every_time():
    name = "Failed case for prompt template " * 10
    [a] = posted([Measure(name=name, score=1.0)])
    [b] = posted([Measure(name=name, score=0.0)])
    assert a["name"] == b["name"]


def test_nul_characters_are_removed_from_texts():
    """Postgres refuses NUL in a text column and would refuse the whole batch."""
    [m] = posted(
        [
            Measure(
                name="a\x00b", description="model said \x00", error="e\x00", score=1.0
            )
        ]
    )
    assert m["name"] == "ab" and m["description"] == "model said " and m["error"] == "e"
