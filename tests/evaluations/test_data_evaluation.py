from a4s_eval.evaluators.data_evaluator import data_evaluator_registry


def test_registration():
    assert len(list(iter(data_evaluator_registry))) >= 1
