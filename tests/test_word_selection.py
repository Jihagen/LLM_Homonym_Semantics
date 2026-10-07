import run_study
from analysis import validate_study_run


def test_study_defaults_exclude_light():
    assert "light" not in run_study.WORDS_DEFAULT
    assert "light" not in validate_study_run.WORDS
    assert len(run_study.WORDS_DEFAULT) == 7
    assert len(validate_study_run.WORDS) == 7
