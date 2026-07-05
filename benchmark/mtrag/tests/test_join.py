from benchmark.mtrag import join


def test_parse_qid_real_format():
    cid, turn = join.parse_qid("e9dd465e8dd63a80dda8f3ce9cba6848<::>1")
    assert cid == "e9dd465e8dd63a80dda8f3ce9cba6848" and turn == 1


def test_split_user_turns():
    text = "|user|:  A?\n|user|: B?\n|user|: C?"
    assert join.split_user_turns(text) == ["A?", "B?", "C?"]


def test_is_standalone():
    assert join.is_standalone(["N/A"]) is True
    assert join.is_standalone(["Follow-up"]) is False
    assert join.is_standalone(["Clarification"]) is False


def test_standalone_index_prefix_match_and_ambiguity():
    convos = [
        {"messages": [
            {"speaker": "user", "text": "What is NAV?", "enrichments": {"Multi-Turn": ["N/A"]}},
            {"speaker": "user", "text": "Why?", "enrichments": {"Multi-Turn": ["Follow-up"]}},
        ]},
    ]
    idx = join.build_standalone_index(convos)
    # turn 1 standalone, turn 2 non-standalone, prefix must match
    qtext = "|user|: What is NAV?\n|user|: Why?"
    assert join.standalone_for(qtext, 1, idx) is True
    assert join.standalone_for(qtext, 2, idx) is False
    # wrong prefix -> None
    assert join.standalone_for("|user|: What is NAV?\n|user|: Different?", 2, idx) is None
