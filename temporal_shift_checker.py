import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ipywidgets as w
from IPython.display import display, clear_output
import sys
sys.path.append("..")
from refdata import month_diff, get_best_valid_time

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 12})

# ── helpers ─────────────────────────────────────────────────────────────────
def month_diff_signed(tgt, src):
    d = (tgt - src) % 12
    return d - 12 if d > 6 else d

def evaluate(s, e, v, buf, n=12):
    row = pd.Series({"start_date": s, "end_date": e, "valid_time": v})
    rows = []
    for m in range(1, 13):
        row["true_valid_time_month"] = v.month
        row["proposed_valid_time_month"] = m
        row["valid_month_shift_backward"] = month_diff(m, v.month)
        row["valid_month_shift_forward"]  = month_diff(v.month, m)
        rows.append([m, get_best_valid_time(row, buf, n)])
    df = pd.DataFrame(rows, columns=["proposed_month", "resulting_valid_time"])
    df["proposed_month_str"] = df["proposed_month"].map(calendar.month_abbr.__getitem__)
    df["acceptable"] = df["resulting_valid_time"].notna()
    return df

def date_picker(label, init):
    lbl = w.Label(label, layout=w.Layout(width="155px"))
    left  = w.Button(icon="chevron-left", layout=w.Layout(width="28px"))
    right = w.Button(icon="chevron-right", layout=w.Layout(width="28px"))
    dp = w.DatePicker(value=init, description="", layout=w.Layout(width="138px"))

    left.on_click (lambda b: dp.value and dp.value.__setattr__("value",
                    (pd.Timestamp(dp.value)-pd.DateOffset(months=1)).to_pydatetime()))
    right.on_click(lambda b: dp.value and dp.value.__setattr__("value",
                    (pd.Timestamp(dp.value)+pd.DateOffset(months=1)).to_pydatetime()))
    return w.HBox([lbl, left, dp, right], layout=w.Layout(align_items="center")), dp
# ── widgets  ────────────────────────────────────────────────────────────────
start_box, start_w = date_picker("Extractions start date", pd.Timestamp("2018-08-01"))
end_box,   end_w   = date_picker("Extractions end date",    pd.Timestamp("2019-11-30"))
valid_box, valid_w = date_picker("True valid time",         pd.Timestamp("2019-06-01"))

buffer_w = w.IntSlider(value=2, min=0, max=6, step=1,
                       description="Buffer (months)",
                       style={'description_width': 'initial'},
                       layout=w.Layout(width="280px"))

radio_sel = w.RadioButtons(layout=w.Layout(width="210px", height="285px", overflow_y="auto"),
                           style={'description_width': '0'})
plot_out  = w.Output()

# ── frames (CSS via ipywidgets layouts only – safer for Voila) ──────────────
date_frame   = w.VBox([w.HTML("<b>Select extraction dates & valid_time</b>"),
                       start_box, end_box, valid_box],
                      layout=w.Layout(border="1px solid #ccc", padding="8px"))
buffer_frame = w.VBox([w.HTML("<b>Buffer settings</b>"), buffer_w],
                      layout=w.Layout(border="1px solid #ccc", padding="8px"))
te_frame     = w.VBox([w.HTML("<b>Temporal Extent (TE) to align</b>"), radio_sel],
                      layout=w.Layout(border="1px solid #ccc", padding="8px"))
plot_frame   = w.VBox([plot_out],
                      layout=w.Layout(border="1px solid #ccc", padding="8px"))

left_col = w.VBox([date_frame, buffer_frame, te_frame], layout=w.Layout(width="330px"))
ui       = w.HBox([left_col, plot_frame])

# ── reactive parts ──────────────────────────────────────────────────────────
current_df = {}

def draw_plot(df, s, e, v, buf):
    with plot_out:
        clear_output()
        fig, ax = plt.subplots(figsize=(8, 3.3))
        days = pd.date_range(s - pd.DateOffset(months=2), e + pd.DateOffset(months=2), freq="D")
        ndvi = 0.4 + 0.35*np.cos((mdates.date2num(days) - mdates.date2num(v))/365.25*2*np.pi)
        ax.plot(days, ndvi, color="forestgreen", label="Simulated NDVI")

        ax.axvspan(s, e, facecolor="skyblue", alpha=.25, edgecolor="skyblue", lw=2,
                   label="Available extractions")
        ax.axvline(v, color="forestgreen", ls="--", lw=1.5, label="True valid_time")

        if radio_sel.value is not None:
            mid_shift = month_diff_signed(df.loc[radio_sel.value, "proposed_month"], v.month)
            mid = v + pd.DateOffset(months=mid_shift)
            st, en = mid - pd.DateOffset(months=5), mid + pd.DateOffset(months=6)
            ax.axvspan(st, en, facecolor="mediumseagreen", alpha=.22, edgecolor="mediumseagreen",
                       lw=2, label="Proposed TE")
            ax.axvspan(st, st+pd.DateOffset(months=buf), facecolor="none",
                       edgecolor="firebrick", hatch='//', lw=0, label="Buffer")
            ax.axvspan(en-pd.DateOffset(months=buf), en, facecolor="none",
                       edgecolor="firebrick", hatch='//', lw=0)
            ax.axvline(mid, color="firebrick", ls="--", lw=1.5, alpha=.7,
                       label="Middle of proposed TE")
            ax.annotate("", xy=(mid, 0.96), xytext=(v, 0.96),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_ylim(0, 1); ax.set_ylabel("NDVI"); ax.set_title("Simulated NDVI")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="upper right")
        plt.tight_layout()
        plt.show()

def redraw(*_):
    s, e, v, buf = start_w.value, end_w.value, valid_w.value, buffer_w.value
    if None in (s, e, v) or not (s < v < e):
        radio_sel.options = []; plot_out.clear_output()
        with plot_out: print("Choose valid dates (start < valid < end)")
        return

    df = evaluate(s, e, v, buf); current_df["df"] = df
    opts = [(f"{row['proposed_month_str']} (✓)" if row['acceptable']
             else f"❌ {row['proposed_month_str']}", idx)
            for idx, row in df.iterrows()]
    radio_sel.options = opts
    radio_sel.value   = radio_sel.value if radio_sel.value in [v for _, v in opts] else None
    draw_plot(df, s, e, v, buf)

radio_sel.observe(lambda c: draw_plot(current_df["df"],
                                      start_w.value, end_w.value,
                                      valid_w.value, buffer_w.value),
                  names="value")

for wdg in (start_w, end_w, valid_w, buffer_w):
    wdg.observe(redraw, names="value")

redraw()
display(ui)