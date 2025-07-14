"""
temporal_shift_checker.py
A self-contained Voilà app – no Jupyter–lab extensions needed.
"""

# ────────────────────────── 1. Imports ──────────────────────────
import calendar
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as w
from IPython.display import display, clear_output

# ---- local helpers (import *after* we add repo root to sys.path) ----
import sys
sys.path.append(str(Path(__file__).parent))        # allow `import refdata`
from refdata import month_diff, get_best_valid_time   # noqa: E402

# ────────────────────────── 2.  helpers  ──────────────────────────
def month_diff_signed(target, source):
    """Shortest signed month shift  (-11…+11)."""
    diff = (target - source) % 12
    return diff - 12 if diff > 6 else diff


def evaluate(s, e, v, buf, n_ts=12):
    row = pd.Series(dict(start_date=s, end_date=e, valid_time=v))
    out = []
    for m in range(1, 13):
        row["true_valid_time_month"] = v.month
        row["proposed_valid_time_month"] = m
        row["valid_month_shift_backward"] = month_diff(m, v.month)
        row["valid_month_shift_forward"] = month_diff(v.month, m)
        out.append([m, get_best_valid_time(row, buf, n_ts)])
    df = pd.DataFrame(out, columns=["proposed_month", "resulting_valid_time"])
    df["proposed_month_str"] = df["proposed_month"].map(calendar.month_abbr.__getitem__)
    df["acceptable"] = df["resulting_valid_time"].notna()
    return df


def date_picker(label, init):
    """One DatePicker flanked by ‹ / › buttons."""
    lab = w.Label(label, layout=w.Layout(width="160px"))
    left = w.Button(icon="chevron-left", layout=w.Layout(width="32px"))
    right = w.Button(icon="chevron-right", layout=w.Layout(width="32px"))
    dp = w.DatePicker(value=init, layout=w.Layout(width="140px"))

    def shift(k):
        if dp.value:
            dp.value = (pd.Timestamp(dp.value) + pd.DateOffset(months=k)).to_pydatetime()

    left.on_click(lambda *_: shift(-1))
    right.on_click(lambda *_: shift(+1))
    return w.HBox([lab, left, dp, right], layout=w.Layout(align_items="center")), dp


# ────────────────────────── 3.  widgets  ──────────────────────────
start_box, start_w = date_picker("Extractions start date", pd.Timestamp("2018-08-01"))
end_box, end_w = date_picker("Extractions end date", pd.Timestamp("2019-11-30"))
valid_box, valid_w = date_picker("True valid time", pd.Timestamp("2019-06-01"))

buffer_w = w.IntSlider(2, 0, 6, 1, description="Buffer (months)",
                       layout=w.Layout(width="280px"),
                       style=dict(description_width="initial"))

# frames
date_frame = w.VBox(
    [w.HTML("<b>Set extraction period & true valid_time</b>"),
     start_box, end_box, valid_box],
    layout=w.Layout(border="1px solid #ccc", padding="10px", margin="5px 0"))

buffer_frame = w.VBox(
    [w.HTML("<b>Buffer settings</b>"), buffer_w],
    layout=w.Layout(border="1px solid #ccc", padding="10px", margin="5px 0"))

# radio selector
radio_sel = w.RadioButtons(
    options=[],
    layout=w.Layout(width="190px", height="300px", overflow_y="auto"),
    style=dict(description_width="0")
)
te_frame = w.VBox([w.HTML("<b>Which temporal extent (TE)?</b>"), radio_sel],
                  layout=w.Layout(border="1px solid #ccc",
                                  padding="10px", margin="5px 0"))

# output area for the plot
plot_out = w.Output()
plot_frame = w.VBox([plot_out],
                    layout=w.Layout(border="1px solid #ccc",
                                    padding="10px", margin="5px 0"))

# ────────────────────────── 4.  drawing routine ──────────────────────────
def draw(df, s, e, vt, buf):
    with plot_out:
        clear_output()
        fig, ax = plt.subplots(figsize=(9, 3.2))

        # NDVI baseline
        days = pd.date_range(s - pd.DateOffset(months=2),
                             e + pd.DateOffset(months=2), freq="D")
        ndvi = 0.4 + 0.35 * np.cos((mdates.date2num(days) -
                                    mdates.date2num(vt)) / 365.25 * 2 * np.pi)
        ax.plot(days, ndvi, color="forestgreen", label="Simulated NDVI")

        ax.axvspan(s, e, color="skyblue", alpha=.25, label="Available extractions")
        ax.axvline(vt, ls="--", color="forestgreen", lw=1.6, label="True valid_time")

        if radio_sel.value is not None:
            row = df.loc[radio_sel.value]
            shift = month_diff_signed(row.proposed_month, vt.month)
            new_mid = vt + pd.DateOffset(months=shift)
            new_start = new_mid - pd.DateOffset(months=5)
            new_end = new_mid + pd.DateOffset(months=6)

            ax.axvspan(new_start, new_end, facecolor="mediumseagreen", alpha=.20,
                       edgecolor="mediumseagreen", lw=2, label="Proposed TE")
            ax.axvspan(new_start, new_start + pd.DateOffset(months=buf),
                       facecolor="none", edgecolor="firebrick",
                       hatch='//', lw=0, alpha=.20, label="Buffer")
            ax.axvspan(new_end - pd.DateOffset(months=buf), new_end,
                       facecolor="none", edgecolor="firebrick",
                       hatch='//', lw=0, alpha=.20)
            ax.axvline(new_mid, ls="--", color="firebrick", lw=1.4,
                       label="Middle of proposed TE")
            ax.annotate("", xy=(new_mid, 0.97), xytext=(vt, 0.97),
                        arrowprops=dict(arrowstyle="->", lw=2, color="black"))

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_ylim(0, 1); ax.set_ylabel("NDVI"); ax.set_title("Simulated NDVI")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


# ────────────────────────── 5.  controller ──────────────────────────
state = dict(df=None)            # tiny mutable cache


def rebuild(*_):
    s, e, vt, buf = start_w.value, end_w.value, valid_w.value, buffer_w.value
    if None in (s, e, vt) or not (s < vt < e):
        radio_sel.options = []
        with plot_out:
            clear_output()
            print("Pick valid start/end/true valid_time first.")
        return

    df = evaluate(pd.Timestamp(s), pd.Timestamp(e), pd.Timestamp(vt), buf)
    state["df"] = df

    radio_sel.options = [
        (f"{row.proposed_month_str} (✓)" if row.acceptable
         else f"❌ {row.proposed_month_str}", idx)
        for idx, row in df.iterrows()
    ]
    draw(df, s, e, vt, buf)


# connect signals *once*
for widget in (start_w, end_w, valid_w, buffer_w):
    widget.observe(rebuild, "value")
radio_sel.observe(lambda c: state["df"] is not None and
                  draw(state["df"], start_w.value, end_w.value, valid_w.value, buffer_w.value),
                  "value")

# ────────────────────────── 6.  show app ──────────────────────────
rebuild()   # first draw
display(w.VBox([w.HTML("<h2>Temporal-shift checker for sample acceptability</h2>"),
                w.HBox([w.VBox([date_frame, buffer_frame, te_frame],
                                layout=w.Layout(width="320px")),
                        plot_frame])]))