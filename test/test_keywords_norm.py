import pytest
from CMEPDA_Exam_repository.exploratory_data_analysis import normalize_keywords

@pytest.mark.parametrize("raw, clean", [
    ("  Higgs Boson  ", "higgs boson"),
    ("Quantum-Field", "quantum field"),
    ("PHYSICS", "physics"),
    ("", ""), # Caso stringa vuota
    ("Neutrino_maSs", "neutrino mass"),
    ("Quark: top", "quark  top")
])
def test_normalize_keywords(raw, clean):
    assert normalize_keywords(raw) == clean
    assert normalize_keywords(raw) == normalize_keywords(normalize_keywords(raw))
