#pip3 install pandas plotly kaleido

import os
import sys
import json
import numpy as np
import plotly
import plotly.express as px

def FDV_cumsum(x):
  y = np.zeros((x.size),dtype=complex)
  y[0] = x[0]
  for i in range(1,x.size):
    y[i] = y[i-1] + x[i]
  return y


def Trapz_CumInt(f_g,dx):
  fgcopy = np.copy(f_g)
  for i in range(0,f_g.size):
    fgcopy[i] = fgcopy[i] - f_g[0]

  CumInteg = 0.5 * dx * (2.0*FDV_cumsum(f_g)-fgcopy)
  return CumInteg


def Trapz_Int(f_g,dx):
  tinteg = 0.5 * dx * (2.0 * np.sum(f_g) - f_g[0] - f_g[f_g.size-1])
  return tinteg


def FT_FDV(f_t,t_g,w_g):
  f_w = np.zeros((w_g.size),dtype=complex)
  for iw in range(0,w_g.size):
    func_tmp = np.exp(ci*w_g[iw]*t_g) * f_t
    f_w[iw] = Trapz_Int(func_tmp,t_g[1]-t_g[0])
  return f_w


if __name__ == "__main__":

    if len(sys.argv) < 3: 
      print("\nUsage: python3 rteom_analysis.py dcdt-file param-file")
      sys.exit()

    dcdt_fp = sys.argv[1]
    param_fp = sys.argv[2]

    if not os.path.exists(dcdt_fp):
      print("dcdt file provided does not exist: " + dcdt_fp)
      exit(0)

    if not os.path.exists(param_fp):
      print("parameter file provided does not exist: " + param_fp)
      exit(0)

    pdata = json.load(open(param_fp))

    delta_real = pdata['delta_real']
    E_Ext_l = pdata['E_Ext_l']
    E_Ext_h = pdata['E_Ext_h']
    dE_local = pdata['dE_local']
    E_c = pdata['E_c']
    E_Corr_0 = pdata['E_Corr_0']
    
    au2ev = 27.2113960
    pi = 4.0*np.arctan(1.0)

    print(pdata)

    # Step 1: read the dcdt file 
    dcdt_res = (open(dcdt_fp).readlines())

    nts_init = len(dcdt_res)
    tsdict = dict()

    for i in range(0,nts_init):
      dcdt_res_i = [float(i) for i in dcdt_res[i].split()]
      tsdict[dcdt_res_i[3]] = dcdt_res[i]

    nts = len(tsdict)
    print("# time steps = %s" %(nts))

    if nts != nts_init:
      print("Elminate duplicates and write to new dcdt file: %s" %(dcdt_fp+".new"))
      dcdt_str = ""
      for k in tsdict.keys():
        dcdt_str += tsdict[k]
      with open(dcdt_fp+".new", "w") as df:
        df.write(dcdt_str)    

    tgrid = np.zeros((nts),dtype=float)
    dcdt = np.zeros((nts),dtype=complex)
    dcdt2 = np.zeros((nts),dtype=complex)

    # delta real imag ts
    for i in range(0,nts):
      dline = tsdict[i+1]
      dcdt_res_i = [float(x) for x in dline.split()]
      tgrid[i] = dcdt_res_i[0]
      dcdt[i] = complex(dcdt_res_i[1],dcdt_res_i[2])

    dT = tgrid[1] - tgrid[0]

    print("delta_t = %s" %(dT))

    nts = nts-4
    tgrid.resize(nts)
    dcdt.resize(nts)
    dcdt2.resize(nts)
    

    # Step 2: Numerical computation of second derivative of the dcdt 
    for i in range(1,nts-1):
      dcdt2[i] = 0.5 * (dcdt[i+1] - dcdt[i-1])/dT

    dcdt2[0] = (-1.0*dcdt[2] + 4*dcdt[1] - 3*dcdt[0])/(2.0*dT)
    dcdt2[nts-1] = dcdt2[nts-2]

    # Step 3: computation of the commulant (C_T) and retarded Green’s function (G_R_t)
    ci = complex(0.0,1.0)
    C_T = ci * Trapz_CumInt(dcdt,dT)
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
        xaxis_title="energy",
        yaxis_title="intensity"
    )    
    # plotly.io.write_image(fig, base_fn+".ft_beta.pdf")
    plotly.io.write_image(fig, base_fn+".ft_gf.png")
    
    iE_QP = np.unravel_index(np.argmax(ft_gf_Ac, axis=None), ft_gf_Ac.shape)[0]
    E_iE_QP  = ft_gf_E[iE_QP]
    Ac_iE_QP = ft_gf_Ac[iE_QP]

    nWdPt = 4
    ft_gf_X = ft_gf_E[iE_QP-nWdPt:iE_QP+nWdPt]
    ft_gf_Y = ft_gf_Ac[iE_QP-nWdPt:iE_QP+nWdPt]

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
      ft_gf_Y1 = ft_gf_Ac[iE_QP-nWdPt]
      ft_gf_Y2 = ft_gf_Ac[iE_QP+nWdPt]
      ft_gf_Pct_Diff = (0.5*(ft_gf_Y1+ft_gf_Y2)/Ac_iE_QP)*100.0


    print("nWdPt = " + str(nWdPt))
    ft_gf_X = ft_gf_E[iE_QP-nWdPt:iE_QP+nWdPt]
    ft_gf_Y = ft_gf_Ac[iE_QP-nWdPt:iE_QP+nWdPt]

    ft_gf_P = np.polyfit(ft_gf_X,1./ft_gf_Y,2)

    # Parameters of Lorentzian
    a0 = -ft_gf_P[1]/(2*ft_gf_P[0])
    a2 = np.sqrt(ft_gf_P[2]/ft_gf_P[0]-a0*a0)
    a1 = 1/(ft_gf_P[0]*a2)

    QP_E_v2 = a0
    Z_QP = a1*pi

    # Check
    if ( abs(QP_E_v2-QP_E_v1) > 0.001*abs(np.mean([QP_E_v1,QP_E_v2])) ):
      print("Large error in estimated E_QP: %s %s\n" %(QP_E_v1, QP_E_v2))
      exit

    E_b = -(QP_E_v2)

    print("\n QP Binding Energy: %s" %(E_b))
    print(" QP Strength:       %s" %(Z_QP))    