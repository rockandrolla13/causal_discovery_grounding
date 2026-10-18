"""Guards on merging method-comparison trees written by separate passes.

The merge exists because the configuration fingerprint covers the method
roster, so a Naturalness-only GPU pass and a competitor-only CPU pass cannot
share an output directory. What these tests lock down is the part a reviewer
would challenge: that a merge cannot quietly produce a comparison in which the
arms were averaged over different conditions, and cannot paper over two passes
that disagree about the same fitted result.

They run in milliseconds on small frames and touch no fitting code.
"""

import json

import pandas as pd
import pytest

from experiments import naturalness_comparison as comparison
from experiments import naturalness_comparison_results as merge


def metrics_row(method, beta=0.0, dgp_seed=20260918, target_site="0", width=0.4, **extra):
    """Return one metrics row shaped like the comparison runner's output.

    Args:
        method: Arm name.
        beta: Assignment strength for the condition.
        dgp_seed: Benchmark draw.
        target_site: Target site label.
        width: Weighted width, used as the agreement quantity.
        **extra: Additional columns, e.g. a pass-specific diagnostic.

    Returns:
        A dictionary carrying every identifying and agreement column.
    """
    row = dict(experiment="toy_comparison", dataset="toy_generated", dgp_seed=dgp_seed,
               seed=20260918, beta=beta, target_site=target_site,
               stage1_backend="tabicl_v2", method=method, n_evaluation=100, n_cells=4,
               weighted_width=width, ate_lower=-0.2, ate_upper=0.2, ate_reference=0.05,
               ate_contains_reference=True, fit_seconds=1.0)
    row.update(extra)
    return row


def write_tree(directory, rows):
    """Write one ``metrics.parquet`` under ``directory`` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "metrics.parquet", index=False)
    return directory


def test_manski_fitted_by_both_passes_collapses_to_one_row(tmp_path):
    """Both passes must fit Manski, and the merge must not double-count it."""
    write_tree(tmp_path / "gpu", [metrics_row("manski"), metrics_row("naturalness", width=0.3)])
    write_tree(tmp_path / "cpu", [metrics_row("manski"), metrics_row("cb_dr", width=0.5)])
    provenance = merge.merge_trees([tmp_path / "gpu", tmp_path / "cpu"], tmp_path / "merged")
    merged = pd.read_parquet(tmp_path / "merged" / "metrics.parquet")
    assert sorted(merged["method"]) == ["cb_dr", "manski", "naturalness"]
    assert provenance["duplicate_rows"] == 1
    assert provenance["conditions_reported"] == 1


def test_passes_that_disagree_on_a_shared_row_abort_the_merge(tmp_path):
    """A differing Manski baseline means the passes did not see the same data."""
    write_tree(tmp_path / "gpu", [metrics_row("manski", width=0.40)])
    write_tree(tmp_path / "cpu", [metrics_row("manski", width=0.55)])
    with pytest.raises(ValueError, match="disagree"):
        merge.merge_trees([tmp_path / "gpu", tmp_path / "cpu"], tmp_path / "merged")


def test_disagreement_can_be_inspected_but_only_deliberately(tmp_path):
    """The override keeps the first row instead of failing, for diagnosis only."""
    write_tree(tmp_path / "gpu", [metrics_row("manski", width=0.40)])
    write_tree(tmp_path / "cpu", [metrics_row("manski", width=0.55)])
    provenance = merge.merge_trees([tmp_path / "gpu", tmp_path / "cpu"], tmp_path / "merged",
                                   allow_disagreement=True)
    assert provenance["disagreeing_rows"] == 1


def test_conditions_one_pass_never_reached_are_not_reported(tmp_path):
    """Naturalness must not average over draws its competitors never saw.

    This is the failure the restriction exists for: the GPU pass covers three
    DGP draws in hours while the CPU pass covers one in days, so the merged
    table would otherwise compare arms over different condition sets.
    """
    write_tree(tmp_path / "gpu", [metrics_row("manski", dgp_seed=d) for d in (1, 2)]
               + [metrics_row("naturalness", dgp_seed=d, width=0.3) for d in (1, 2)])
    write_tree(tmp_path / "cpu", [metrics_row("manski", dgp_seed=1),
                                  metrics_row("cb_dr", dgp_seed=1, width=0.5)])
    provenance = merge.merge_trees([tmp_path / "gpu", tmp_path / "cpu"], tmp_path / "merged")
    reported = pd.read_parquet(tmp_path / "merged" / "metrics.parquet")
    everything = pd.read_parquet(tmp_path / "merged" / "metrics_all.parquet")
    assert set(reported["dgp_seed"]) == {1}
    assert set(everything["dgp_seed"]) == {1, 2}
    assert provenance["conditions_dropped"] == 1
    dropped = pd.read_csv(tmp_path / "merged" / "dropped_conditions.csv")
    assert dropped.loc[0, "missing_methods"] == "cb_dr"


def test_keeping_everything_is_possible_but_not_the_default(tmp_path):
    """``--keep-all`` exists for inspection and must be asked for explicitly."""
    write_tree(tmp_path / "gpu", [metrics_row("manski", dgp_seed=d) for d in (1, 2)]
               + [metrics_row("naturalness", dgp_seed=d) for d in (1, 2)])
    write_tree(tmp_path / "cpu", [metrics_row("manski", dgp_seed=1),
                                  metrics_row("cb_dr", dgp_seed=1)])
    merge.merge_trees([tmp_path / "gpu", tmp_path / "cpu"], tmp_path / "merged", keep_all=True)
    reported = pd.read_parquet(tmp_path / "merged" / "metrics.parquet")
    assert set(reported["dgp_seed"]) == {1, 2}


def test_each_passes_own_diagnostics_survive_the_merge(tmp_path):
    """The passes record different diagnostics; both belong in the merged row."""
    write_tree(tmp_path / "gpu", [metrics_row("manski", balke_pearl_infeasible_cells=2)])
    write_tree(tmp_path / "cpu", [metrics_row("manski", blearner_oracle_gamma=1.7)])
    merge.merge_trees([tmp_path / "gpu", tmp_path / "cpu"], tmp_path / "merged")
    merged = pd.read_parquet(tmp_path / "merged" / "metrics.parquet")
    assert merged.loc[0, "balke_pearl_infeasible_cells"] == 2
    assert merged.loc[0, "blearner_oracle_gamma"] == pytest.approx(1.7)


def test_leaf_metadata_reads_the_condition_and_the_run_identity(tmp_path):
    """A leaf's path gives its condition; the nearest ``identity.json`` its run."""
    root = tmp_path / "cpu" / "toy_generated"
    leaf = root / "toy_generated" / "dgp20260918" / "seed20260918" / "beta0.5" / "target3"
    leaf.mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps(
        dict(fingerprint="abc", config=dict(experiment_name="toy_comparison", stage1_backend="tabicl_v2"))))

    recovered = merge._leaf_metadata(leaf)
    assert recovered["dataset"] == "toy_generated"
    assert recovered["dgp_seed"] == 20260918
    assert recovered["beta"] == pytest.approx(0.5)
    assert recovered["target_site"] == "3"
    assert recovered["experiment"] == "toy_comparison"
    assert recovered["stage1_backend"] == "tabicl_v2"


def test_a_metrics_table_without_the_identifying_columns_is_rejected(tmp_path):
    """A stray parquet must fail loudly rather than merge into a wrong key."""
    directory = tmp_path / "bad"
    directory.mkdir()
    pd.DataFrame([dict(method="manski", weighted_width=0.3)]).to_parquet(
        directory / "metrics.parquet", index=False)
    with pytest.raises(ValueError, match="identifying columns"):
        merge.load_metrics([directory])


def test_an_empty_results_root_fails_rather_than_reporting_nothing(tmp_path):
    """Silently reporting an empty comparison would be worse than an error."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        merge.load_metrics([tmp_path / "empty"])


def test_an_incomplete_merge_exits_distinctly_from_a_broken_one(tmp_path):
    """The split runner treats status 3 as incomplete and status 1 as broken."""
    write_tree(tmp_path / "gpu", [metrics_row("manski", dgp_seed=d) for d in (1, 2)]
               + [metrics_row("naturalness", dgp_seed=d) for d in (1, 2)])
    write_tree(tmp_path / "cpu", [metrics_row("manski", dgp_seed=1),
                                  metrics_row("cb_dr", dgp_seed=1)])
    status = merge.main(["merge", str(tmp_path / "gpu"), str(tmp_path / "cpu"),
                         "--output-dir", str(tmp_path / "merged")])
    assert status == merge.DROPPED_CONDITIONS_EXIT == 3


def test_a_complete_merge_exits_zero(tmp_path):
    """No condition dropped means the merged table is fully comparable."""
    write_tree(tmp_path / "gpu", [metrics_row("manski"), metrics_row("naturalness")])
    write_tree(tmp_path / "cpu", [metrics_row("manski"), metrics_row("cb_dr")])
    assert merge.main(["merge", str(tmp_path / "gpu"), str(tmp_path / "cpu"),
                       "--output-dir", str(tmp_path / "merged")]) == 0


def test_a_dataset_with_no_dgp_draw_still_merges(tmp_path):
    """Twins has a single fixed draw, so ``dgp_seed`` is absent everywhere.

    The absence arrives as ``None`` from a metrics table and as ``NaN`` once a
    frame has been round-tripped, and the two must key identically or the
    common-condition restriction would drop every Twins condition.
    """
    write_tree(tmp_path / "gpu", [metrics_row("manski", dgp_seed=None),
                                  metrics_row("naturalness", dgp_seed=None)])
    write_tree(tmp_path / "cpu", [metrics_row("manski", dgp_seed=float("nan")),
                                  metrics_row("cb_dr", dgp_seed=float("nan"))])
    provenance = merge.merge_trees([tmp_path / "gpu", tmp_path / "cpu"], tmp_path / "merged")
    assert provenance["conditions_reported"] == 1
    assert provenance["conditions_dropped"] == 0
    assert sorted(pd.read_parquet(tmp_path / "merged" / "metrics.parquet")["method"]) == [
        "cb_dr", "manski", "naturalness"]


def test_leaf_metadata_parses_a_dataset_with_no_dgp_draw(tmp_path):
    """The Twins leaf layout writes ``dgpNone``, which must parse, not crash."""
    root = tmp_path / "cpu" / "seed20260918-targets012"
    leaf = root / "twins_reduced" / "dgpNone" / "seed20260918" / "beta0.25" / "target1"
    leaf.mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps(
        dict(config=dict(experiment_name="exp28", stage1_backend="tabicl_v2"))))

    recovered = merge._leaf_metadata(leaf)
    assert recovered["dataset"] == "twins_reduced"
    assert recovered["dgp_seed"] is None
    assert recovered["beta"] == pytest.approx(0.25)
    assert recovered["experiment"] == "exp28"


def write_choice_leaf(root, beta, target_site, choices):
    """Write one condition leaf carrying a ``selected_choices.parquet``.

    Args:
        root: Run output directory holding ``identity.json``.
        beta: Assignment strength naming the ``beta`` path component.
        target_site: Target site naming the ``target`` path component.
        choices: Rows for the screening table. An empty list writes the empty
            frame a condition gets when no candidate satisfied the predicates.

    Returns:
        The leaf directory that was written.
    """
    leaf = root / "toy_generated" / "dgp20260918" / "seed20260918" / f"beta{beta}" / f"target{target_site}"
    leaf.mkdir(parents=True, exist_ok=True)
    (root / "identity.json").write_text(json.dumps(
        dict(fingerprint="abc", config=dict(experiment_name="toy_comparison", stage1_backend="tabicl_v2"))))
    columns = ["is_primary_choice", "covariate", "two_c", "i_deviation"]
    pd.DataFrame(choices, columns=columns).to_parquet(leaf / "selected_choices.parquet",
                                                      index=False)
    return leaf


def test_an_empty_screening_table_is_read_as_the_trigger_not_firing(tmp_path):
    """No ``(j, epsilon, tau)`` satisfied the predicates, so nothing was discovered."""
    root = tmp_path / "gpu" / "run"
    write_choice_leaf(root, 0.5, "3", [])
    write_choice_leaf(root, 1.0, "3", [dict(is_primary_choice=True, covariate="mager8",
                                            two_c=0.006, i_deviation=0.004)])
    status = merge.load_trigger_status([tmp_path / "gpu"]).sort_values("beta")
    assert list(status["trigger_fired"]) == [False, True]
    assert list(status["tolerance_covers_deviation"]) == [False, True]
    assert status.iloc[1]["selected_covariate"] == "mager8"


def test_a_tolerance_below_the_selected_candidates_deviation_is_not_covering(tmp_path):
    """``2c`` under the candidate's own deviation leaves the LP no feasible point."""
    root = tmp_path / "gpu" / "run"
    write_choice_leaf(root, 0.5, "3", [dict(is_primary_choice=True, covariate="birmon",
                                            two_c=0.0108, i_deviation=0.0124)])
    status = merge.load_trigger_status([tmp_path / "gpu"])
    assert bool(status.iloc[0]["trigger_fired"]) is True
    assert bool(status.iloc[0]["tolerance_covers_deviation"]) is False


def test_filtering_on_the_trigger_keeps_every_method_on_a_surviving_condition(tmp_path):
    """The competitors must be averaged over exactly the conditions Naturalness is."""
    root = tmp_path / "gpu" / "run"
    write_choice_leaf(root, 0.5, "3", [])
    write_choice_leaf(root, 1.0, "3", [dict(is_primary_choice=True, covariate="mager8",
                                            two_c=0.006, i_deviation=0.004)])
    rows = [metrics_row(method, beta=beta, target_site="3")
            for beta in (0.5, 1.0) for method in ("manski", "naturalness", "cb_dr")]
    pd.DataFrame(rows).to_parquet(root / "metrics.parquet", index=False)

    provenance = merge.merge_trees([tmp_path / "gpu"], tmp_path / "merged",
                                   naturalness_trigger="fired")
    reported = pd.read_parquet(tmp_path / "merged" / "metrics.parquet")
    assert provenance["conditions_reported"] == 1
    assert provenance["conditions_excluded_by_trigger"] == 1
    assert sorted(reported["method"]) == ["cb_dr", "manski", "naturalness"]
    assert list(reported["beta"].unique()) == [1.0]
    excluded = pd.read_csv(tmp_path / "merged" / "excluded_conditions.csv")
    assert excluded.iloc[0]["excluded_because"] == "discovery selected no candidate"


def test_the_consistent_mode_also_drops_an_uncovered_tolerance(tmp_path):
    """``fired`` keeps a condition the stricter mode must exclude."""
    root = tmp_path / "gpu" / "run"
    write_choice_leaf(root, 0.5, "3", [dict(is_primary_choice=True, covariate="birmon",
                                            two_c=0.0108, i_deviation=0.0124)])
    write_choice_leaf(root, 1.0, "3", [dict(is_primary_choice=True, covariate="mager8",
                                            two_c=0.006, i_deviation=0.004)])
    rows = [metrics_row(method, beta=beta, target_site="3")
            for beta in (0.5, 1.0) for method in ("manski", "naturalness")]
    pd.DataFrame(rows).to_parquet(root / "metrics.parquet", index=False)

    fired = merge.merge_trees([tmp_path / "gpu"], tmp_path / "fired",
                              naturalness_trigger="fired")
    strict = merge.merge_trees([tmp_path / "gpu"], tmp_path / "strict",
                               naturalness_trigger="consistent")
    assert fired["conditions_reported"] == 2
    assert strict["conditions_reported"] == 1
    excluded = pd.read_csv(tmp_path / "strict" / "excluded_conditions.csv")
    assert excluded.iloc[0]["excluded_because"] == \
        "2c below the selected candidate's measured deviation"


def test_restricting_without_any_screening_output_fails_loudly(tmp_path):
    """A comparator-only tree cannot be filtered on a trigger it never recorded."""
    write_tree(tmp_path / "cpu", [metrics_row("manski"), metrics_row("cb_dr")])
    with pytest.raises(ValueError, match="selected_choices.parquet"):
        merge.merge_trees([tmp_path / "cpu"], tmp_path / "merged",
                          naturalness_trigger="fired")


def test_the_default_merge_keeps_every_condition(tmp_path):
    """Filtering is opt-in, so an unqualified merge must not silently restrict."""
    root = tmp_path / "gpu" / "run"
    write_choice_leaf(root, 0.5, "3", [])
    rows = [metrics_row(method, beta=0.5, target_site="3")
            for method in ("manski", "naturalness")]
    pd.DataFrame(rows).to_parquet(root / "metrics.parquet", index=False)
    provenance = merge.merge_trees([tmp_path / "gpu"], tmp_path / "merged")
    assert provenance["conditions_reported"] == 1
    assert provenance["naturalness_trigger"] == "any"
    assert not (tmp_path / "merged" / "excluded_conditions.csv").exists()


def test_the_coverage_note_is_one_line_when_every_arm_shares_its_seeds():
    """Equal coverage needs no per-arm detail, only the band and the seed count."""
    rows = [metrics_row(method, seed=seed) for seed in (1, 2, 3)
            for method in ("manski", "naturalness", "cb_dr")]
    assert merge._coverage_note(pd.DataFrame(rows)) == (
        "shaded band: 90% CI of the mean over conditions (3 screening seeds per method)")


def test_the_coverage_note_names_every_arm_a_deeper_pass_extended():
    """Extra seeds deepen a whole pass, so no single arm may be credited with them.

    The Naturalness pass also fits Manski and Balke-Pearl, so adding seeds to it
    leaves three arms at the deeper count. Naming only the first of those in
    alphabetical order would credit Balke-Pearl with the extension.
    """
    rows = [metrics_row(method, seed=seed) for seed in (1, 2, 3)
            for method in ("manski", "naturalness", "balke_pearl", "cb_dr", "cb_ipw")]
    rows += [metrics_row(method, seed=seed) for seed in range(4, 21)
             for method in ("manski", "naturalness", "balke_pearl")]
    band, detail = merge._coverage_note(pd.DataFrame(rows)).split("\n")
    assert band == "shaded band: 90% CI of the mean over conditions"
    assert detail == ("20 screening seeds for Manski, Naturalness, Balke-Pearl; "
                      "3 for every other arm")


def test_report_methods_share_the_exp25_colours_and_labels():
    """Every reported arm is drawn with the colour and label Exp25 gives it."""
    from experiments import exp25_hit_ratio_plots as exp25_plots

    for method, model in merge.EXP25_MODEL_FOR_METHOD.items():
        assert merge.METHOD_COLORS[method] == exp25_plots.MODEL_COLORS[model]
        assert merge.METHOD_LABELS[method] == exp25_plots.MODEL_LABELS[model]
        assert merge.EXP28_OPTIMAL_METHOD_LABELS[method] == exp25_plots.MODEL_LABELS[model]
    assert merge.METHOD_COLORS["blearner_oracle"] == exp25_plots.MODEL_COLORS[
        "blearner[dgp_oracle]"]
    # The runner's own labels are left alone: its source is fingerprinted.
    assert comparison.METHOD_LABELS["balke_pearl"] == "Balke-Pearl IV"


def _pre_cap_tree(root):
    """Write a one-condition tree as the runner wrote it before the Manski cap.

    The B-Learner arm spans the whole effect range in both cells, which is
    twice the Manski width, and is scored uncapped exactly as the runner used
    to score it.

    Args:
        root: Directory to hold the shard.

    Returns:
        The shard directory.
    """
    shard = root / "shard"
    leaf = shard / "twins_reduced" / "dgpNone" / "seed1" / "beta1" / "target0"
    leaf.mkdir(parents=True)
    (shard / "identity.json").write_text(json.dumps({"config": dict(
        experiment_name="exp28_twins_method_comparison", stage1_backend="tabicl_v2",
        twins_covariates=["z"], twins_bias_covariate="b")}))
    baseline = pd.DataFrame(dict(z=[0, 1], lower=[-.4, -.6], upper=[.6, .4], n_fit=[40, 40],
                                 n_fit_arm0=[20, 20], n_fit_arm1=[20, 20],
                                 common_supported=[True, True]))
    truth = pd.DataFrame(dict(z=[0, 1], n_evaluation=[50, 50], reference_effect=[.0, .0],
                              paired_effect=[.0, .0], evaluation_weight=[.5, .5]))
    raw = pd.DataFrame(dict(z=[0, 1], lower=[-1., -1.], upper=[1., 1.], status="optimal"))
    raw.to_parquet(leaf / "blearner_oracle_raw.parquet", index=False)
    frames = {"manski": baseline.assign(used_manski_fallback=False, status="manski"),
              "blearner_oracle": comparison._complete_bounds(raw, baseline, ("z",))}
    condition = dict(experiment="exp28_twins_method_comparison", dataset="twins_reduced",
                     dgp_seed=None, seed=1, beta=1.0, target_site="0", stage1_backend="tabicl_v2")
    scored, metrics = [], []
    for method, frame in frames.items():
        rows, metric = comparison.score_bounds(frame, truth, ("z",), method)
        scored.append(rows)
        metrics.append(dict(metric, **condition, gamma_exp_beta=2.7))
    pd.concat(scored, ignore_index=True).to_parquet(leaf / "bounds_scored.parquet", index=False)
    (leaf / "completion.json").write_text(json.dumps(dict(metrics=metrics, errors=[],
                                                          status="complete")))
    pd.DataFrame(metrics).to_parquet(shard / "metrics.parquet", index=False)
    return shard


def test_capping_an_old_tree_rescores_the_blearner_and_leaves_the_source(tmp_path):
    """The copy's B-Learner is cut back to Manski; the source is never touched."""
    source = _pre_cap_tree(tmp_path / "source")
    before = (source / "metrics.parquet").read_bytes()
    provenance = merge.cap_blearner_tree(tmp_path / "source", tmp_path / "capped")
    assert (source / "metrics.parquet").read_bytes() == before
    metrics = pd.read_parquet(tmp_path / "capped" / "shard" / "metrics.parquet").set_index("method")
    assert metrics.loc["blearner_oracle", "weighted_width"] == pytest.approx(1.0)
    assert metrics.loc["manski", "weighted_width"] == pytest.approx(1.0)
    # Diagnostics the runner attached to the row survive the re-scoring.
    assert metrics.loc["blearner_oracle", "gamma_exp_beta"] == pytest.approx(2.7)
    assert provenance["arms"] == {"blearner_oracle": dict(conditions=1, cells=2,
                                                          cells_capped=2, cells_disjoint=0)}
    leaf = tmp_path / "capped" / "shard" / "twins_reduced" / "dgpNone" / "seed1" / "beta1" / "target0"
    scored = pd.read_parquet(leaf / "bounds_scored.parquet")
    capped = scored.loc[scored.method.eq("blearner_oracle")]
    assert capped.width.tolist() == pytest.approx([1.0, 1.0])
    assert capped.uncapped_upper.tolist() == pytest.approx([1.0, 1.0])
    completion = json.loads((leaf / "completion.json").read_text())
    assert {m["method"]: m["weighted_width"] for m in completion["metrics"]}["blearner_oracle"] \
        == pytest.approx(1.0)


def test_capping_refuses_an_existing_output_and_any_already_capped_tree(tmp_path):
    """The cap is applied once, into a fresh directory, and never on top of itself."""
    _pre_cap_tree(tmp_path / "source")
    merge.cap_blearner_tree(tmp_path / "source", tmp_path / "capped")
    with pytest.raises(FileExistsError):
        merge.cap_blearner_tree(tmp_path / "source", tmp_path / "capped")
    # A capped copy keeps the uncapped fit beside capped scores, so a second
    # pass fails the check that the stored scores are the uncapped ones.
    with pytest.raises(ValueError, match="does not reproduce"):
        merge.cap_blearner_tree(tmp_path / "capped", tmp_path / "capped_twice")
    # A tree the current runner wrote carries the cap in its raw predictions.
    leaf = (tmp_path / "source" / "shard" / "twins_reduced" / "dgpNone" / "seed1" / "beta1"
            / "target0")
    raw = pd.read_parquet(leaf / "blearner_oracle_raw.parquet")
    baseline = pd.read_parquet(leaf / "bounds_scored.parquet").query("method == 'manski'")
    comparison.intersect_with_manski(raw, baseline, ("z",)).to_parquet(
        leaf / "blearner_oracle_raw.parquet", index=False)
    with pytest.raises(ValueError, match="already capped"):
        merge.cap_blearner_tree(tmp_path / "source", tmp_path / "from_new_runner")
