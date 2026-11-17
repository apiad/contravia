from caiman.kept import kept


def test_kept_builds_once_per_stamp_and_stays_out_of_the_state_dump():
    import json

    state, calls = {}, []

    def build():
        calls.append(1)
        return [len(calls)]

    assert kept(state, "x", 1, build) == [1]
    assert kept(state, "x", 1, build) == [1]
    assert kept(state, "x", 2, build) == [2]   # a saved source file changes the stamp
    assert len(calls) == 2
    assert json.dumps(state, default=repr) == '{"kept": "<kept: x>"}'
