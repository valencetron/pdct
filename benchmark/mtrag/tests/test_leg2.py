from benchmark.mtrag import run_mtrag


def test_slice_key():
    assert run_mtrag.slice_key(False, 7) == ("non_standalone", "late")
    assert run_mtrag.slice_key(True, 1) == ("standalone", "early")
    assert run_mtrag.slice_key(False, 4) == ("non_standalone", "late")
    assert run_mtrag.slice_key(True, 3) == ("standalone", "early")
