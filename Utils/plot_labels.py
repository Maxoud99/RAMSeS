"""Detector names in figures: when to SHORTEN them, and how to say so.

Detector names are long everywhere by default — the pool name is the upstream's
own (`OmniAnomaly_1`, not `OA_1`), and that is what the run page, the
explanation cards, the comprehensive report and every figure carry. Nothing has
to be expanded on the way out.

Two figures are the exception, and only two. Both draw one label per detector
against a crowded axis, where a sixteen-character name either collides with its
neighbour or pushes the legend over the plot:

    * the ranking-score trace       (Thompson Sampling: Ranking)
    * both expected-reward traces   (Thompson Sampling: Selection — and up to
                                     107 legend entries)

Those call `abbreviate_detector` on the label and `draw_abbreviation_key` under
the figure, which says what the short form stands for. Everywhere else the name
is simply used.

The key is drawn only when something in `names` actually shortens, so a pool of
LOF/HBOS/MCD gets no empty box.
"""

from Utils.pipeline_spec import abbreviation_legend

__all__ = ["abbreviation_key_text", "draw_abbreviation_key"]


def abbreviation_key_text(names) -> str:
    """'SR: SpectralResidual' pairs, one per shortened family. '' if none.

    Families rather than instances: SR_1..SR_4 all shorten the same way, so
    four lines saying so is three lines of noise.
    """
    legend = abbreviation_legend(names)
    families = {}
    for short, long in legend.items():
        families.setdefault(short.split("_")[0], long.split("_")[0])
    if not families:
        return ""
    pairs = ", ".join(f"{short}: {long}" for short, long in sorted(families.items()))
    return f"Abbreviations — {pairs}"


def draw_abbreviation_key(fig, names, y: float = -0.02) -> bool:
    """Write the key under `fig`. Returns whether anything was drawn.

    Placed on the FIGURE rather than the axes: both callers put their legend
    outside the axes on the right, and a note anchored to the axes would either
    sit under the legend or be clipped by `bbox_inches="tight"`.
    """
    text = abbreviation_key_text(names)
    if not text:
        return False
    fig.text(0.5, y, text, ha="center", fontsize=8, color="dimgrey")
    return True
