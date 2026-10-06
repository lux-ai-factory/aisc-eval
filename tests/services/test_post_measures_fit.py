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
    with patch("requests.post", return_value=Mock(status_code=201, headers={}, text="")) as post:
        post_measures(EVALUATION, PLUGIN_RUN, measures)
    return post.call_args.kwargs["json"][str(PLUGIN_RUN)]


def test_texts_longer_than_the_engine_stores_are_clipped_with_an_ellipsis():
    long = "Considering everything you know about {GROUP}, " * 12
    [m] = posted([Measure(name=long, description=long, unit=long, error=long, score=1.0,
                          dimensions={"template": long})])
    for field in ("name", "description", "unit", "error"):
        assert len(m[field]) <= 255 and m[field].endswith("…"), field
    assert m["dimensions"]["template"] == long and m["score"] == 1.0


def test_short_and_missing_texts_are_untouched():
    [m] = posted([Measure(name="Pass rate", description="ok", score=0.5)])
    assert m["name"] == "Pass rate" and m["description"] == "ok" and m["unit"] is None and m["error"] is None
