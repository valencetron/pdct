from benchmark.mtrag import keyphrase


def test_corpus_extractor_filters_stopwords():
    docs = ["a roth ira is a retirement account with tax free withdrawals",
            "mortgage refinance interest rate and closing costs explained",
            "market capitalization versus net asset value comparison",
            "dividend yield and payout ratio for income investors"]
    ex = keyphrase.CorpusExtractor(docs, top_k=4)
    cs = ex.extract(docs[0])
    assert cs and all(c == c.lower() for c in cs)
    assert "the" not in cs and "is" not in cs


def test_extract_query_concepts_single():
    cs = keyphrase.extract_query_concepts("Which is more important, market cap or NAV?")
    assert isinstance(cs, list)


def test_slugify():
    assert keyphrase.slugify("Interest Rate!") == "interest-rate"
