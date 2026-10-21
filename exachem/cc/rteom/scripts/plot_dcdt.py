#pip3 install plotly kaleido

import os
import sys
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


if __name__ == "__main__":

    if len(sys.argv) < 2:
      print("\nUsage: python3 plot_dcdt.py dcdt-file")
      sys.exit()

    dcdt_fp = sys.argv[1]

    if not os.path.exists(dcdt_fp):
      print("dcdt file provided does not exist: " + dcdt_fp)
      exit(0)

    # Step 1: read the dcdt file
    dcdt_lines = [dl for dl in open(dcdt_fp).readlines() if dl.strip()]
    nts_init = len(dcdt_lines)

    if nts_init == 0:
      print("dcdt file is empty: " + dcdt_fp)
      exit(0)

    dcdt_cols = [[float(x) for x in dl.split()] for dl in dcdt_lines]

    if len(dcdt_cols[0]) < 4:
      print("dcdt file has %s columns, 4 expected: %s" %(len(dcdt_cols[0]), dcdt_fp))
      exit(0)

    # Two column layouts are in circulation:
    #   ts time real imag  (rt_eom_cd_ccsd)
    #   time real imag ts  (rt_eom_cd_cc2)
    # The time step counter is the integer valued column with no zero entry.
    def is_ts_column(icol):
      col = [c[icol] for c in dcdt_cols]
      return all(float(v).is_integer() for v in col) and min(col) >= 1.0

    if is_ts_column(0):
      i_ts, i_t, i_re, i_im = 0, 1, 2, 3
    elif is_ts_column(3):
      i_ts, i_t, i_re, i_im = 3, 0, 1, 2
    else:
      print("Could not identify the time step column in " + dcdt_fp)
      exit(0)

    tgrid = np.array([row[i_t] for row in dcdt_cols], dtype=float)
    dre   = np.array([row[i_re] for row in dcdt_cols], dtype=float)
    dim   = np.array([row[i_im] for row in dcdt_cols], dtype=float)

    base_fn = os.path.basename(dcdt_fp)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("time vs real", "time vs imaginary"))
    fig.add_trace(go.Scatter(x=tgrid, y=dre, mode="lines", name="real"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tgrid, y=dim, mode="lines", name="imaginary"), row=1, col=2)

    fig.update_xaxes(title_text="time", row=1, col=1)
    fig.update_xaxes(title_text="time", row=1, col=2)
    fig.update_yaxes(title_text="real", row=1, col=1)
    fig.update_yaxes(title_text="imaginary", row=1, col=2)

    fig.update_layout(
        font_family="Courier New",
        title_font_family="Times New Roman",
        title_font_color="red",
        title=base_fn,
        showlegend=False,
    )

    out_png = base_fn + ".re_im_vs_t.png"
    try:
      plotly.io.write_image(fig, out_png, width=1200, height=500)
      print("Wrote " + out_png)
    except Exception as e:
      print("Skipping plot, could not write " + out_png + ": %s" %(e))
