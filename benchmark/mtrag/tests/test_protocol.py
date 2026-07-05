from benchmark.mtrag import protocol


def test_constructed_arms_equal_middle_len():
    convos = [
        {"messages": [{"speaker": "user", "text": "q1"}, {"speaker": "user", "text": "q2"}, {"speaker": "user", "text": "q3"}]},
        {"messages": [{"speaker": "user", "text": "r1"}, {"speaker": "user", "text": "r2"}, {"speaker": "user", "text": "r3"}]},
    ]
    arms = protocol.construct_arms(convos, "opener", "closer", max_middle=2)
    assert arms["A"][0] == "opener" and arms["A"][-1] == "closer"
    assert arms["C"] == ["closer"]
    assert len(arms["A"]) == len(arms["B"])
    assert arms["A"][1:-1] != arms["B"][1:-1]
