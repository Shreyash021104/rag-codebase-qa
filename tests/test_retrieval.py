"""Tests for hybrid retrieval: tokenization, rank fusion, and the BM25 cache.

These are the parts that decide which chunks reach the model, and none of them
needed a database — they only appeared untestable because the fusion was
inlined into a DB-bound function.
"""

import app.retrieval as retrieval
from app.retrieval import (
    RRF_K,
    rank_fused,
    reciprocal_rank_fusion,
    tokenize,
)


class TestTokenize:
    def test_camel_case_yields_whole_identifier_and_parts(self):
        assert tokenize("markPresent") == ["markpresent", "mark", "present"]

    def test_snake_case_yields_whole_identifier_and_parts(self):
        assert tokenize("mark_present") == ["mark_present", "mark", "present"]

    def test_plain_word_is_not_duplicated(self):
        # A single word has no parts to split, so emitting it twice would
        # double its BM25 term frequency and quietly skew scoring.
        assert tokenize("present") == ["present"]

    def test_consecutive_capitals_split_before_the_final_word(self):
        assert tokenize("HTTPServer") == ["httpserver", "http", "server"]

    def test_punctuation_and_syntax_are_dropped(self):
        assert tokenize("def get_conn(self) -> None:") == [
            "def", "get_conn", "get", "conn", "self", "none",
        ]

    def test_a_query_for_a_part_can_match_the_whole_identifier(self):
        # The entire point of code-aware tokenization: "present" must overlap
        # the tokens produced for `markPresent`.
        assert set(tokenize("present")) & set(tokenize("markPresent"))


class TestReciprocalRankFusion:
    def test_a_document_found_by_both_legs_outranks_either_leg_alone(self):
        vector = [10, 20, 30]
        bm25 = [40, 20, 50]
        fused = reciprocal_rank_fusion([vector, bm25])
        # 20 is 2nd in both; 10 and 40 are 1st in one leg and absent from the other.
        assert rank_fused(fused, 1) == [20]

    def test_score_matches_the_rrf_formula(self):
        fused = reciprocal_rank_fusion([[7]], k=RRF_K)
        assert fused[7] == 1.0 / (RRF_K + 1)

    def test_scores_from_each_list_add(self):
        fused = reciprocal_rank_fusion([[7], [7]])
        assert fused[7] == 2.0 / (RRF_K + 1)

    def test_rank_is_all_that_matters_not_list_length(self):
        # A short list and a long list contribute identically at equal rank,
        # which is what lets two incompatible scoring scales be combined.
        short = reciprocal_rank_fusion([[1]])
        long = reciprocal_rank_fusion([[1, 2, 3, 4, 5, 6]])
        assert short[1] == long[1]

    def test_empty_legs_are_harmless(self):
        assert reciprocal_rank_fusion([[], []]) == {}
        assert reciprocal_rank_fusion([[1, 2], []]) == reciprocal_rank_fusion([[1, 2]])

    def test_result_is_the_union_of_both_legs(self):
        fused = reciprocal_rank_fusion([[1, 2], [3, 4]])
        assert set(fused) == {1, 2, 3, 4}


class TestRankFused:
    def test_orders_by_descending_score(self):
        assert rank_fused({1: 0.1, 2: 0.9, 3: 0.5}, 3) == [2, 3, 1]

    def test_ties_break_on_id_so_results_are_deterministic(self):
        # Without an explicit tie-break the order would follow dict insertion,
        # i.e. whichever retrieval leg happened to be iterated first.
        assert rank_fused({5: 0.5, 3: 0.5, 9: 0.5}, 3) == [3, 5, 9]
        assert rank_fused({9: 0.5, 3: 0.5, 5: 0.5}, 3) == [3, 5, 9]

    def test_respects_the_limit(self):
        assert rank_fused({1: 0.9, 2: 0.8, 3: 0.7}, 2) == [1, 2]


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeConn:
    """Stands in for psycopg: repos lookups return a version, chunk lookups a corpus."""

    def __init__(self, version="2026-01-01", chunk_count=3):
        self.version = version
        self.chunk_count = chunk_count
        self.corpus_reads = 0

    def execute(self, sql, params=None):
        if "FROM repos" in sql:
            return FakeCursor([(self.version,)])
        self.corpus_reads += 1
        repo_id = params[0]
        return FakeCursor(
            [
                (repo_id * 1000 + i, f"file{i}.py", f"sym{i}", f"def sym{i}(): pass")
                for i in range(self.chunk_count)
            ]
        )


class TestBM25Cache:
    def setup_method(self):
        retrieval._bm25_cache.clear()

    def test_a_repo_is_indexed_once_and_then_served_from_cache(self):
        conn = FakeConn()
        retrieval._bm25_for_repo(conn, 1)
        retrieval._bm25_for_repo(conn, 1)
        assert conn.corpus_reads == 1

    def test_reindexing_invalidates_the_cached_index(self):
        conn = FakeConn(version="v1")
        retrieval._bm25_for_repo(conn, 1)
        conn.version = "v2"  # repo re-indexed
        retrieval._bm25_for_repo(conn, 1)
        assert conn.corpus_reads == 2

    def test_the_cache_is_bounded(self, monkeypatch):
        # Each entry holds a whole repo's tokenized corpus. Unbounded, a server
        # queried across many repos grows until it is killed.
        monkeypatch.setattr(retrieval, "_BM25_CACHE_SIZE", 3)
        conn = FakeConn()
        for repo_id in range(1, 11):
            retrieval._bm25_for_repo(conn, repo_id)
        assert len(retrieval._bm25_cache) == 3

    def test_eviction_is_least_recently_used(self, monkeypatch):
        monkeypatch.setattr(retrieval, "_BM25_CACHE_SIZE", 2)
        conn = FakeConn()
        retrieval._bm25_for_repo(conn, 1)
        retrieval._bm25_for_repo(conn, 2)
        retrieval._bm25_for_repo(conn, 1)  # repo 1 is now the most recent
        retrieval._bm25_for_repo(conn, 3)  # evicts repo 2, not repo 1
        assert set(retrieval._bm25_cache) == {1, 3}

    def test_a_repo_with_no_chunks_is_not_cached(self):
        conn = FakeConn(chunk_count=0)
        assert retrieval._bm25_for_repo(conn, 1) is None
        assert retrieval._bm25_cache == {}
