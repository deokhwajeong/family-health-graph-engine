from backend.app.anomaly_detection import PatternCorrelationAnalyzer
from backend.app.graph_embeddings import SimpleGraphSAGE, extract_day_features
from backend.app.synthetic_data import (
    DEMO_END_DATE,
    DEMO_MEMBERS,
    build_demo_family_household,
    build_single_child_household,
)

FICTIONAL_NAMES = {"Mia", "Leo", "Alex", "Jordan"}


def _nodes_of(G, node_type):
    return [n for n, a in G.nodes(data=True) if a.get("type") == node_type]


def test_demo_family_has_four_fictional_members():
    G = build_demo_family_household(num_days=14)
    people = _nodes_of(G, "person")
    assert {G.nodes[p]["name"] for p in people} == FICTIONAL_NAMES
    assert len(_nodes_of(G, "household")) == 1
    assert len(_nodes_of(G, "day")) == 14 * len(DEMO_MEMBERS)


def test_window_ends_on_demo_end_date():
    G = build_demo_family_household(num_days=30)
    dates = sorted(G.nodes[d]["date"] for d in _nodes_of(G, "day"))
    assert dates[-1] == DEMO_END_DATE


def test_same_day_is_identical_across_window_sizes():
    short = build_demo_family_household(num_days=7)
    long = build_demo_family_household(num_days=60)
    assert extract_day_features(short, "person:mia", DEMO_END_DATE) == extract_day_features(
        long, "person:mia", DEMO_END_DATE
    )


def test_day_features_include_linked_metrics():
    G = build_demo_family_household(num_days=7)
    f = extract_day_features(G, "person:leo", DEMO_END_DATE)
    for key in ("total_calories", "sleep_quality", "metric_energy_level", "metric_steps",
                "metric_resting_heart_rate"):
        assert key in f, key


def test_cross_day_patterns_are_discoverable():
    G = build_demo_family_household(num_days=90)
    analyzer = PatternCorrelationAnalyzer(lag_days=1)
    corr = analyzer.analyze(G, "person:alex")
    assert corr["sleep_quality->metric_focus_score(+1d)"] > 0.5
    assert corr["total_sugar_g->sleep_quality(+1d)"] < -0.4


def test_single_child_graph_and_embeddings():
    G = build_single_child_household(num_days=10)
    assert [G.nodes[p]["name"] for p in _nodes_of(G, "person")] == ["Leo"]
    sage = SimpleGraphSAGE(embedding_dim=16, num_aggregation_layers=2)
    emb = sage.fit_embeddings(G)
    assert len(emb) == G.number_of_nodes()
    assert sage.get_daily_embedding(G, "person:leo", DEMO_END_DATE) is not None
