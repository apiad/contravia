from contravia.baked import baked, stamp


def test_a_result_is_computed_once_and_again_when_its_data_changes(tmp_path):
    data = tmp_path / "city.npz"
    data.write_bytes(b"one")
    calls = []

    def build():
        calls.append(1)
        return {"n": len(calls)}

    keep = tmp_path / "city.npz.thing.pkl"
    assert baked(keep, stamp(data), build) == {"n": 1}
    assert baked(keep, stamp(data), build) == {"n": 1}
    data.write_bytes(b"two, longer")
    assert baked(keep, stamp(data), build) == {"n": 2}
    assert len(calls) == 2


def test_a_broken_file_is_computed_again(tmp_path):
    keep = tmp_path / "x.pkl"
    keep.write_bytes(b"not a pickle")
    assert baked(keep, "k", lambda: 7) == 7
    assert baked(keep, "k", lambda: 8) == 7
