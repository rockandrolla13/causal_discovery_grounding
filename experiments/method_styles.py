"""One registry of method display styles: labels, colours, markers and drawing order.

Every figure that draws a method takes its look from here, so a method has the same
label and colour in every figure. Two key spaces exist:

- *Models* are the Exp25 analysis keys (``naturalness``, ``blearner[dgp_oracle]``, ...)
  drawn by :mod:`experiments.exp25_hit_ratio_plots`.
- *Methods* are the naturalness-comparison roster keys (``blearner_oracle``, ``cb_pi``, ...)
  drawn by :mod:`experiments.naturalness_comparison_results`.
  :data:`EXP25_MODEL_FOR_METHOD` maps a roster method to the model whose label and
  colour it shares.

Markers are the one per-figure override: the roster figures keep their own
:data:`METHOD_MARKERS` (for example Balke-Pearl is ``"P"`` there and ``"X"`` in
:data:`MODEL_MARKERS`), so each figure family draws exactly as before.

This module imports no experiment script.
"""

from __future__ import annotations

# Matplotlib mathtext symbols (no LaTeX needed), held as constants so f-strings that
# also contain escapes such as a newline can interpolate them.
BETA_SYMBOL = r"$\beta$"
GAMMA_SYMBOL = r"$\Gamma$"

# Exp25 analysis models.
MODEL_LABELS = {
    "naturalness": "Naturalness",
    "blearner[dgp_oracle]": f"B-Learner ({GAMMA_SYMBOL} dgp-oracle)",
    "blearner[oracle]": f"B-Learner ({GAMMA_SYMBOL} oracle)",
    "blearner[exp_beta]": f"B-Learner ({GAMMA_SYMBOL} exp-{BETA_SYMBOL})",
    "cb_ipw": "CB-IPW",
    "cb_dr": "CB-DR",
    "balke_pearl": "Balke-Pearl",
    "manski": "Manski",
    "transport": "Transport",
}
MODEL_SHORT_LABELS = {
    "naturalness": "Nat",
    "blearner[dgp_oracle]": "BL dgp",
    "blearner[oracle]": "BL orc",
    "blearner[exp_beta]": "BL exp",
    "cb_ipw": "IPW",
    "cb_dr": "DR",
    "balke_pearl": "BP",
    "manski": "Manski",
    "transport": "Trans",
}
MODEL_COLORS = {
    "naturalness": "tab:purple",
    "blearner[dgp_oracle]": "tab:red",
    "blearner[oracle]": "#7f1d1d",
    "blearner[exp_beta]": "#f4977a",
    "cb_ipw": "tab:blue",
    "cb_dr": "tab:green",
    # Exp28 draws Balke-Pearl violet, too close to Naturalness here.
    "balke_pearl": "tab:orange",
    "manski": "#64748b",
    # Colour-blind distinguishable from Naturalness and Balke-Pearl (worst pairwise Delta E 14.4).
    "transport": "tab:cyan",
}
MODEL_MARKERS = {
    "naturalness": "o",
    "blearner[dgp_oracle]": "s",
    "blearner[oracle]": "D",
    "blearner[exp_beta]": "^",
    "cb_ipw": "v",
    "cb_dr": "P",
    "balke_pearl": "X",
    "manski": "d",
    "transport": "h",
}
FALLBACK_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")
FALLBACK_COLORS = (
    "tab:orange",
    "tab:brown",
    "tab:olive",
    "tab:cyan",
    "tab:pink",
    "0.45",
)
# Manuscript figures are drawn at the printed width (inches) with PAPER_RC's print-sized
# TrueType fonts, so LaTeX includes them at about natural size.
PAPER_FIGURE_WIDTH = 6.5
PAPER_RC = {
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "figure.labelsize": 10.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Naturalness-comparison roster methods.
#: Drawing order. Manski first because it is the reference rather than a rival.
METHOD_ORDER = ("manski", "naturalness", "balke_pearl", "blearner_exp_beta",
                "blearner_oracle", "cb_pi", "cb_ra", "cb_ipw", "cb_dr")

#: Exp25 analysis model each roster method is drawn as, for shared colours and
#: labels; ``blearner_oracle`` takes Gamma from the declared latent propensity.
EXP25_MODEL_FOR_METHOD = {
    "manski": "manski",
    "naturalness": "naturalness",
    "balke_pearl": "balke_pearl",
    "blearner_exp_beta": "blearner[exp_beta]",
    "blearner_oracle": "blearner[dgp_oracle]",
    "cb_ipw": "cb_ipw",
    "cb_dr": "cb_dr",
}

#: Colours from the Exp25 palette; ``cb_pi`` and ``cb_ra`` have their own so a
#: run that requests them through ``--meta-learners`` is still drawn.
METHOD_COLORS = {
    "cb_pi": "#0e7490",
    "cb_ra": "#15803d",
    **{method: MODEL_COLORS[model] for method, model in EXP25_MODEL_FOR_METHOD.items()},
}

METHOD_MARKERS = {
    "manski": "_", "naturalness": "o", "balke_pearl": "P",
    "blearner_exp_beta": "s", "blearner_oracle": "D",
    "cb_pi": "^", "cb_ra": "v", "cb_ipw": "<", "cb_dr": ">",
}
