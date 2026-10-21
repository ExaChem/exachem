# pip3 install numpy plotly kaleido
#
# Usage: python3 rteom_analysis.py <prefix>.dcdt params.json
#
# Required entries in the JSON parameter file:
#   "delta_real", "E_Ext_l", "E_Ext_h", "dE_local", "E_c", "E_Corr_0"
#
#   delta_real       : broadening parameter (au)
#   E_Ext_l, E_Ext_h : energy extensions (au) below and above the reference
#                      energy E_c + E_Corr_0
#   dE_local         : energy-grid spacing (au)
#   E_c              : core orbital energy (au), i.e. the orbital energy of the
#                      pcore orbital in the closed-shell reference SCF
#   E_Corr_0         : N-electron ground-state correlation energy (au), from a
#                      separate CCSD run on the closed-shell reference
#   The spectrum is reported on the energy grid shifted by E_c + E_Corr_0
#   and converted to eV.
#
# Optional entries to enable linear-prediction Padé analysis:
#   "use_pade": true,
#   "pade_order": 800,
#   "pade_reg": 1.0e-8,
#   "pade_eta": 0.0,
#   "pade_nfit": 0,
#   "pade_root_radius": 1.000001,
#   "pade_dt": 0.05,       # same as rt_step_max
#   "trim_last_points": 0

import os
import sys
import json
import numpy as np
import plotly
import plotly.express as px

def Trapz_CumInt(f_g, t_g):
  """Cumulative trapezoidal integral on a possibly nonuniform time grid."""
  f_g = np.asarray(f_g, dtype=complex)
  t_g = np.asarray(t_g, dtype=float)
  if f_g.size != t_g.size:
    raise ValueError("f_g and t_g must have the same size")

  CumInteg = np.zeros_like(f_g, dtype=complex)
  dt = np.diff(t_g)
  CumInteg[1:] = np.cumsum(0.5 * (f_g[:-1] + f_g[1:]) * dt)
  return CumInteg


def Trapz_Int(f_g, t_g):
  """Trapezoidal integral on a possibly nonuniform time grid."""
  f_g = np.asarray(f_g, dtype=complex)
  t_g = np.asarray(t_g, dtype=float)
  if f_g.size != t_g.size:
    raise ValueError("f_g and t_g must have the same size")

  return np.sum(0.5 * (f_g[:-1] + f_g[1:]) * np.diff(t_g))


def FT_FDV(f_t,t_g,w_g):
  f_w = np.zeros((w_g.size),dtype=complex)
  for iw in range(0,w_g.size):
    func_tmp = np.exp(1j*w_g[iw]*t_g) * f_t
    f_w[iw] = Trapz_Int(func_tmp,t_g)
  return f_w


def Read_DCDT(dcdt_fp):
  """Read C++ output: iteration, time, real(dC/dt), imag(dC/dt).

  If a restarted calculation appended duplicate iteration numbers, the last
  occurrence is retained. Returned points are ordered by iteration number.
  """
  records = {}
  raw_count = 0

  with open(dcdt_fp, "r") as dcdt_stream:
    for line_number, line in enumerate(dcdt_stream, start=1):
      stripped = line.strip()
      if not stripped or stripped.startswith("#"):
        continue

      fields = stripped.split()
      if len(fields) < 4:
        raise ValueError(
          "Malformed dcdt line %d: expected at least four columns" % line_number
        )

      try:
        iteration_value = float(fields[0])
        iteration = int(iteration_value)
        time_au = float(fields[1])
        value = complex(float(fields[2]), float(fields[3]))
      except ValueError as exc:
        raise ValueError("Malformed numeric data on dcdt line %d" % line_number) from exc

      if not np.isclose(iteration_value, iteration, rtol=0.0, atol=1.0e-12):
        raise ValueError("Non-integer iteration number on dcdt line %d" % line_number)

      records[iteration] = (time_au, value, stripped)
      raw_count += 1

  if len(records) < 3:
    raise ValueError("At least three unique dcdt points are required")

  iterations = sorted(records)
  tgrid = np.array([records[i][0] for i in iterations], dtype=float)
  dcdt = np.array([records[i][1] for i in iterations], dtype=complex)

  dt = np.diff(tgrid)
  if not np.all(np.isfinite(tgrid)) or not np.all(np.isfinite(dcdt)):
    raise ValueError("The dcdt file contains non-finite values")
  if np.any(dt <= 0.0):
    raise ValueError("dcdt time points must be strictly increasing")

  if len(records) != raw_count:
    new_fp = dcdt_fp + ".new"
    print("Eliminate duplicates and write to new dcdt file: %s" % new_fp)
    with open(new_fp, "w") as output_stream:
      for iteration in iterations:
        output_stream.write(records[iteration][2] + "\n")

  return tgrid, dcdt, np.asarray(iterations, dtype=int)


def Resample_Uniform(f_t, t_g, dt):
  """Linearly resample a complex signal for linear-prediction Padé."""
  if dt <= 0.0:
    raise ValueError("pade_dt must be positive")
  if t_g.size < 2:
    raise ValueError("At least two time points are required for resampling")

  n_uniform = int(np.floor((t_g[-1] - t_g[0]) / dt)) + 1
  if n_uniform < 6:
    raise ValueError("The uniform Padé grid must contain at least six points")

  t_uniform = t_g[0] + dt * np.arange(n_uniform, dtype=float)
  f_uniform = (
    np.interp(t_uniform, t_g, f_t.real)
    + 1j * np.interp(t_uniform, t_g, f_t.imag)
  )
  return t_uniform, f_uniform


def Linear_Prediction_Pade(
    f_t, dt, w_g, order=200, reg=1.0e-8, eta=0.0,
    nfit=None, root_radius=1.000001
):
  """Evaluate an infinite-time pole expansion obtained by linear prediction.

  The fitted signal is represented as

      f[n] = sum_k amplitude[k] * root[k]**n,

  and analytically transformed from t=0 to infinity. ``w_g`` may be complex;
  a positive imaginary part supplies the usual retarded broadening. ``eta`` is
  an optional *additional* damping applied before fitting.
  """
  f_t = np.asarray(f_t, dtype=complex)
  w_g = np.asarray(w_g, dtype=complex)
  nfull = f_t.size

  if dt <= 0.0:
    raise ValueError("Padé analysis requires dt > 0")
  if nfit is None or nfit <= 0 or nfit > nfull:
    nfit = nfull

  g = np.array(f_t[:nfit], dtype=complex, copy=True)
  if eta < 0.0:
    raise ValueError("pade_eta must be non-negative")
  if eta > 0.0:
    g *= np.exp(-eta * np.arange(g.size) * dt)

  n = g.size
  if n < 6:
    raise ValueError("At least 6 fitting points are required for Padé analysis")

  order = int(order)
  if order < 2:
    raise ValueError("pade_order must be at least 2")
  if order >= n // 2:
    order = max(2, n // 3)
    print("Padé order was too large for nfit; reset to %d" % order)

  rows = n - order
  print(
    "Padé fitting: nfit=%d, order=%d, rows=%d, reg=%e, eta=%e"
    % (n, order, rows, reg, eta)
  )

  # g[n] + a[0]g[n-1] + ... + a[M-1]g[n-M] = 0
  windows = np.lib.stride_tricks.sliding_window_view(g, order + 1)
  A = windows[:, :-1][:, ::-1].copy()
  b = -windows[:, -1]

  # Solve the Tikhonov-regularized least-squares problem without forming A^H A.
  if reg < 0.0:
    raise ValueError("pade_reg must be non-negative")
  if reg > 0.0:
    A_aug = np.vstack((A, np.sqrt(reg) * np.eye(order, dtype=complex)))
    b_aug = np.concatenate((b, np.zeros(order, dtype=complex)))
    coeff, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
  else:
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)

  # z^M + a[0]z^(M-1) + ... + a[M-1] = 0
  roots = np.roots(np.concatenate(([1.0 + 0.0j], coeff)))
  finite = np.isfinite(roots)
  roots = roots[finite]

  if root_radius <= 0.0:
    raise ValueError("pade_root_radius must be positive")
  roots = roots[np.abs(roots) <= root_radius]
  if roots.size == 0:
    raise RuntimeError(
      "No stable Padé roots found; try a smaller pade_order, a larger "
      "pade_reg, or a slightly larger pade_root_radius"
    )

  print("Padé stable roots kept: %d" % roots.size)

  # Refit amplitudes for the retained stable poles.
  powers = np.arange(n, dtype=int)[:, None]
  V = roots[None, :] ** powers
  amplitudes, *_ = np.linalg.lstsq(V, g, rcond=None)
  reconstruction = V @ amplitudes
  relative_error = np.linalg.norm(reconstruction - g) / max(
    np.linalg.norm(g), np.finfo(float).eps
  )
  print("Padé relative reconstruction error: %.6e" % relative_error)

  # Infinite-time transform. The -g[0]/2 term gives trapezoidal endpoint
  # weighting, consistent with FT_FDV.
  f_w = np.zeros(w_g.size, dtype=complex)
  exp_iwdt = np.exp(1j * w_g * dt)
  for amplitude, root in zip(amplitudes, roots):
    ratio = root * exp_iwdt
    if np.any(np.abs(ratio) >= 1.0):
      raise RuntimeError(
        "Padé series does not converge on the requested grid; increase the "
        "positive frequency broadening or reduce pade_root_radius"
      )
    denominator = 1.0 - ratio
    if np.any(np.abs(denominator) < 1.0e-14):
      raise RuntimeError(
        "Padé pole lies on the requested frequency grid; use positive broadening"
      )
    f_w += amplitude / denominator
  f_w = dt * (f_w - 0.5 * g[0])
  return f_w, roots, amplitudes, relative_error


if __name__ == "__main__":

    if len(sys.argv) < 3: 
      print("\nUsage: python3 rteom_analysis.py dcdt-file param-file")
      sys.exit(1)

    dcdt_fp = sys.argv[1]
    param_fp = sys.argv[2]

    if not os.path.exists(dcdt_fp):
      print("dcdt file provided does not exist: " + dcdt_fp)
      sys.exit(1)

    if not os.path.exists(param_fp):
      print("parameter file provided does not exist: " + param_fp)
      sys.exit(1)

    pdata = json.load(open(param_fp))

    delta_real = pdata['delta_real']
    E_Ext_l = pdata['E_Ext_l']
    E_Ext_h = pdata['E_Ext_h']
    dE_local = pdata['dE_local']
    E_c = pdata['E_c']
    E_Corr_0 = pdata['E_Corr_0']

    # Optional linear-prediction Padé settings.
    use_pade = bool(pdata.get('use_pade', False))
    pade_order = int(pdata.get('pade_order', 200))
    pade_reg = float(pdata.get('pade_reg', 1.0e-8))
    pade_eta = float(pdata.get('pade_eta', 0.0))
    pade_nfit = int(pdata.get('pade_nfit', 0))
    pade_root_radius = float(pdata.get('pade_root_radius', 1.000001))
    pade_dt = float(pdata.get('pade_dt', 0.05))
    trim_last_points = int(pdata.get('trim_last_points', 0))

    if delta_real < 0.0:
      raise ValueError("delta_real must be non-negative")
    if dE_local <= 0.0:
      raise ValueError("dE_local must be positive")
    if pade_dt <= 0.0:
      raise ValueError("pade_dt must be positive")
    if trim_last_points < 0:
      raise ValueError("trim_last_points must be non-negative")
    
    au2ev = 27.2113960
    pi = 4.0*np.arctan(1.0)

    print(pdata)

    # Step 1: read iteration, time, real(dC/dt), imag(dC/dt).
    tgrid, dcdt, iteration_grid = Read_DCDT(dcdt_fp)

    if trim_last_points > 0:
      if tgrid.size - trim_last_points < 3:
        raise ValueError("Too few points remain after trim_last_points")
      tgrid = tgrid[:-trim_last_points].copy()
      dcdt = dcdt[:-trim_last_points].copy()
      iteration_grid = iteration_grid[:-trim_last_points].copy()

    nts = tgrid.size
    time_steps = np.diff(tgrid)
    print("# time steps = %s" % nts)
    print("minimum delta_t = %.15g" % np.min(time_steps))
    print("maximum delta_t = %.15g" % np.max(time_steps))
    print("mean delta_t    = %.15g" % np.mean(time_steps))

    # Step 2: derivative of dC/dt on the nonuniform time grid.
    dcdt2 = np.gradient(dcdt, tgrid, edge_order=2)

    # Step 3: computation of the commulant (C_T) and retarded Green’s function (G_R_t)
    ci = complex(0.0,1.0)
    C_T = ci * Trapz_CumInt(dcdt,tgrid)
    G_R_t = -1.0 * ci * np.exp(C_T)

    base_fn = os.path.basename(dcdt_fp)

    with open(base_fn+".cumulant_gf.txt", 'w') as f:
      astr = ""
      for i in range(0,nts):
        astr += str(tgrid[i]) + "\t" + str(C_T[i].real) + "\t" + str(C_T[i].imag) + "\t" + str(G_R_t[i].real) + "\t" + str(G_R_t[i].imag) + "\n"
      f.write(astr)

    # Step 4
    delta_Local = delta_real*ci
    E_mn_Local = -1.0*E_Ext_l
    E_mx_Local = E_Ext_h
    nE_local = int((E_mx_Local - E_mn_Local)/dE_local+1)

    E_Grid_Local = np.zeros((nE_local),dtype=complex)
    for iE in range(0,nE_local):
      E_Grid_Local[iE] = E_mn_Local + iE*dE_local + delta_Local

    G_R_w  = FT_FDV(G_R_t, tgrid, E_Grid_Local)
    Beta_w = FT_FDV(-1.0*ci*dcdt2, tgrid, E_Grid_Local)

    print("nE_local = " + str(nE_local))

    ft_gf_E = np.zeros((nE_local),dtype=float)
    ft_gf_Ac = np.zeros((nE_local),dtype=float)
    with open(base_fn+".ft_greens_function.txt", 'w') as f:
      astr = ""
      for i in range(0,nE_local):
        ft_gf_E[i] = au2ev * (E_c + E_Corr_0 + (E_Grid_Local[i]).real) 
        ft_gf_Ac[i] = ((-G_R_w[i].imag)/pi)/au2ev
        astr += str(ft_gf_E[i]) + "\t" + str(ft_gf_Ac[i]) + "\n"
      f.write(astr)

    if use_pade:
      print("Running linear-prediction Padé analysis for G_R_t ...")
      if abs(tgrid[0]) > 1.0e-10:
        raise ValueError(
          "Padé analysis requires the dcdt trajectory to begin at time zero"
        )

      tgrid_pade, G_R_t_pade = Resample_Uniform(G_R_t, tgrid, pade_dt)
      print("Padé uniform delta_t = %.15g" % pade_dt)
      print("Padé uniform points  = %d" % tgrid_pade.size)

      with open(base_fn+".pade_uniform_gf_t.txt", 'w') as f:
        for time_value, gf_value in zip(tgrid_pade, G_R_t_pade):
          f.write(
            str(time_value) + "\t" + str(gf_value.real) + "\t"
            + str(gf_value.imag) + "\n"
          )

      G_R_w_pade, pade_roots, pade_amp, pade_error = Linear_Prediction_Pade(
        G_R_t_pade,
        pade_dt,
        E_Grid_Local,
        order=pade_order,
        reg=pade_reg,
        eta=pade_eta,
        nfit=(None if pade_nfit <= 0 else pade_nfit),
        root_radius=pade_root_radius,
      )

      pade_gf_E = np.zeros((nE_local), dtype=float)
      pade_gf_Ac = np.zeros((nE_local), dtype=float)
      with open(base_fn+".pade_greens_function.txt", 'w') as f:
        for i in range(0, nE_local):
          pade_gf_E[i] = au2ev * (
            E_c + E_Corr_0 + E_Grid_Local[i].real
          )
          pade_gf_Ac[i] = ((-G_R_w_pade[i].imag)/pi)/au2ev
          f.write(str(pade_gf_E[i]) + "\t" + str(pade_gf_Ac[i]) + "\n")

      fig_pade = px.line(x=pade_gf_E, y=pade_gf_Ac, title="pade_gf")
      fig_pade.update_layout(
        font_family="Courier New",
        title_font_family="Times New Roman",
        title_font_color="red",
        title=base_fn+" Padé",
        xaxis_title="Energy (eV)",
        yaxis_title="Spectral function (1/eV)"
      )
      plotly.io.write_image(fig_pade, base_fn+".pade_gf.png")

      iE_QP_pade = int(np.argmax(pade_gf_Ac))
      print("\n Padé QP Binding Energy estimate: %s" % (-pade_gf_E[iE_QP_pade]))
      print(" Padé peak intensity:            %s" % pade_gf_Ac[iE_QP_pade])
      print(" Padé reconstruction error:      %s" % pade_error)
      print(" Padé number of retained poles:  %s" % pade_roots.size)
      print(" Padé output:                    %s" % (base_fn+".pade_greens_function.txt"))

    ft_b_x = np.zeros((nE_local),dtype=float) #energy
    ft_b_y = np.zeros((nE_local),dtype=float) #intensity
    with open(base_fn+".ft_beta.txt", 'w') as f:
      astr = ""
      for i in range(0,nE_local):
        ft_b_x[i] = E_Grid_Local[i].real
        ft_b_y[i] = Beta_w[i].real/pi
        astr += str(ft_b_x[i]) + "\t" + str(ft_b_y[i]) + "\n"
      f.write(astr)

    with open(base_fn+".dc2dt2.txt", 'w') as f:
      astr = ""
      for i in range(0,nts):
        tmp_val = -1.0*ci*dcdt2[i]
        astr += str(tgrid[i]) + "\t" + str(tmp_val.real) + "\t" + str(tmp_val.imag) + "\n"
      f.write(astr)

    fig = px.line(x=ft_gf_E, y=ft_gf_Ac, title="ft_gf")
    fig.update_layout(
        font_family="Courier New",
        title_font_family="Times New Roman",
        title_font_color="red",

        title=base_fn,
        xaxis_title="Energy (eV)",
        yaxis_title="Spectral function (1/eV)"
    )    
    # plotly.io.write_image(fig, base_fn+".ft_beta.pdf")
    plotly.io.write_image(fig, base_fn+".ft_gf.png")
    
    iE_QP = np.unravel_index(np.argmax(ft_gf_Ac, axis=None), ft_gf_Ac.shape)[0]
    E_iE_QP  = ft_gf_E[iE_QP]
    Ac_iE_QP = ft_gf_Ac[iE_QP]

    if Ac_iE_QP <= 0.0:
      raise RuntimeError("The maximum spectral intensity is not positive")
    if iE_QP < 4 or iE_QP + 4 >= nE_local:
      raise RuntimeError("The QP peak is too close to an energy-grid boundary")

    nWdPt = 4
    ft_gf_X = ft_gf_E[iE_QP-nWdPt:iE_QP+nWdPt+1]
    ft_gf_Y = ft_gf_Ac[iE_QP-nWdPt:iE_QP+nWdPt+1]

    ft_gf_P = np.polyfit(ft_gf_X,ft_gf_Y,2)
    ft_gf_PP = np.polyder(ft_gf_P)
    QP_E_v1 = np.roots(ft_gf_PP)[0]

    # Try using a way to fit a Lorentzian
    nWdPt = 1
    ft_gf_Y1 = ft_gf_Ac[iE_QP-nWdPt]
    ft_gf_Y2 = ft_gf_Ac[iE_QP+nWdPt]
    ft_gf_Pct_Diff = (0.5*(ft_gf_Y1+ft_gf_Y2)/Ac_iE_QP)*100.0

    while ( ft_gf_Pct_Diff >= 30.0 ):
      nWdPt = nWdPt+1
      if iE_QP-nWdPt < 0 or iE_QP+nWdPt >= nE_local:
        raise RuntimeError("Lorentzian fitting window reached the energy-grid boundary")
      ft_gf_Y1 = ft_gf_Ac[iE_QP-nWdPt]
      ft_gf_Y2 = ft_gf_Ac[iE_QP+nWdPt]
      ft_gf_Pct_Diff = (0.5*(ft_gf_Y1+ft_gf_Y2)/Ac_iE_QP)*100.0


    print("nWdPt = " + str(nWdPt))
    ft_gf_X = ft_gf_E[iE_QP-nWdPt:iE_QP+nWdPt+1]
    ft_gf_Y = ft_gf_Ac[iE_QP-nWdPt:iE_QP+nWdPt+1]

    if np.any(ft_gf_Y <= 0.0):
      raise RuntimeError("Lorentzian fitting window contains non-positive intensity")

    ft_gf_P = np.polyfit(ft_gf_X,1./ft_gf_Y,2)

    # Parameters of Lorentzian
    a0 = -ft_gf_P[1]/(2*ft_gf_P[0])
    gamma_squared = ft_gf_P[2]/ft_gf_P[0]-a0*a0
    if ft_gf_P[0] <= 0.0 or gamma_squared <= 0.0:
      raise RuntimeError("The fitted peak is not a valid positive Lorentzian")
    a2 = np.sqrt(gamma_squared)
    a1 = 1/(ft_gf_P[0]*a2)

    QP_E_v2 = a0
    Z_QP = a1*pi

    # Check
    if ( abs(QP_E_v2-QP_E_v1) > 0.001*abs(np.mean([QP_E_v1,QP_E_v2])) ):
      print("Large error in estimated E_QP: %s %s\n" %(QP_E_v1, QP_E_v2))
      sys.exit(1)

    E_b = -(QP_E_v2)

    print("\n QP Binding Energy: %s" %(E_b))
    print(" QP Strength:       %s" %(Z_QP))    
