"""Shared matplotlib style for the thesis figures."""

import matplotlib.pyplot as plt

# Full text width of the thesis in inches.
TEXT_WIDTH = 6.3

# Reference palette: categorical slots 1-2, blue<->red diverging, neutral midpoint.
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
NEUTRAL, INK, INK_2, MUTED, GRID = "#f0efec", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SPINE = "#c3c2b7"


def apply_rc():
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42})


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=INK_2, labelsize=7)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def style_axes_3d(ax):
    """3D counterpart of `style_axes`: transparent panes, light grid."""
    for name, set_label in (
        ("x", ax.set_xlabel),
        ("y", ax.set_ylabel),
        ("z", ax.set_zlabel),
    ):
        set_label(f"${name}$", fontsize=7, labelpad=-10, color=INK_2)
    ax.tick_params(labelsize=5, pad=-3, colors=MUTED)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = GRID
        axis._axinfo["grid"]["linewidth"] = 0.4
