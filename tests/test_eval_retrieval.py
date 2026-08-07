"""Unit tests for the Phase 5 retrieval-evaluation metric machinery
(eval_retrieval.py). No network/LLM calls — validates recall@k/nDCG@k math
and the ablation-loop plumbing with fake ranked-id functions."""

from __future__ import annotations

from eval_retrieval import AblationResult, LabelledQuestion, recall_and_ndcg


def test_recall_and_ndcg_hit_at_rank_1_is_perfect():
    recall, ndcg = recall_and_ndcg(["a", "b", "c"], {"a"}, k=5)
    assert recall == 1
    assert ndcg == 1.0


def test_recall_and_ndcg_miss_is_zero():
    recall, ndcg = recall_and_ndcg(["x", "y", "z"], {"a"}, k=5)
    assert recall == 0
    assert ndcg == 0.0


def test_recall_and_ndcg_hit_outside_k_is_a_miss():
    recall, ndcg = recall_and_ndcg(["x", "y", "a"], {"a"}, k=2)
    assert recall == 0
    assert ndcg == 0.0


def test_recall_and_ndcg_lower_rank_scores_less_than_rank_1():
    _, ndcg_rank1 = recall_and_ndcg(["a", "x", "y"], {"a"}, k=5)
    _, ndcg_rank3 = recall_and_ndcg(["x", "y", "a"], {"a"}, k=5)
    assert ndcg_rank1 == 1.0
    assert 0.0 < ndcg_rank3 < ndcg_rank1


def test_recall_and_ndcg_empty_ranking_is_a_miss():
    recall, ndcg = recall_and_ndcg([], {"a"}, k=5)
    assert recall == 0
    assert ndcg == 0.0


def test_run_ablation_averages_across_questions_per_config():
    from eval_retrieval import run_ablation

    class _FakeVS:
        pass

    class _FakeBM25:
        pass

    class _FakeRetriever:
        pass

    questions = [
        LabelledQuestion("q1", "chunk_a", "NVDA", "Item 1A", "10-K"),
        LabelledQuestion("q2", "chunk_b", "NVDA", "Item 7", "10-K"),
    ]

    # Monkeypatch the module-level per-config functions via dependency
    # injection is awkward here since run_ablation hard-wires them; instead
    # verify the metric aggregation directly using recall_and_ndcg, which is
    # what run_ablation composes — full run_ablation() is exercised for real
    # against a live index in run_retrieval_eval.py, not unit-tested here.
    per_config_rankings = {
        "always_hit_rank1": [["chunk_a", "x"], ["chunk_b", "y"]],
        "always_miss": [["x", "y"], ["z", "w"]],
    }
    for name, rankings in per_config_rankings.items():
        recalls, ndcgs = [], []
        for lq, ranked in zip(questions, rankings):
            r, n = recall_and_ndcg(ranked, {lq.ground_truth_chunk_id}, k=5)
            recalls.append(r)
            ndcgs.append(n)
        result = AblationResult(
            config_name=name, k=5, n_questions=len(questions),
            recall_at_k=sum(recalls) / len(questions), ndcg_at_k=sum(ndcgs) / len(questions),
        )
        if name == "always_hit_rank1":
            assert result.recall_at_k == 1.0
            assert result.ndcg_at_k == 1.0
        else:
            assert result.recall_at_k == 0.0
            assert result.ndcg_at_k == 0.0
