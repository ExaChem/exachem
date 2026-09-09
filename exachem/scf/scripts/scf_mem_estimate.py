#!/usr/bin/env python3
"""
Estimate node memory for an ExaChem SCF/HF run.

Two kinds of memory are modelled:

  * REPLICATED -- dense Eigen matrices in exachem/scf/ held in full on a rank.
    Per-rank cost does NOT fall as ranks/nodes are added. Multiplied by
    ranks-per-node to get the node cost.

  * DISTRIBUTED -- the TAMM tensors (Fock/density family, DIIS history, the DF
    3-index tensor, ...). There is no per-rank figure; the tensor has one global
    size that is spread over every rank in the job. Per node it costs
    (global size / number of nodes).

    node total = replicated_per_node + distributed_global / n_nodes

Only exachem/scf/ is modelled: the replicated EigenTensors members plus the
sibling Matrix locals in the driver, and the persistent TAMMTensors members on
the distributed side.

Assumptions:
  * Northo == nbf_orig == nbf == N (no linear dependence, spherical basis).
  * SCF-iteration residency, plus the one transient big enough to matter: the DF
    3c build buffer xyZ (N*N*ndf), which coexists with xyK during 3c-init and is
    folded into the distributed TOTAL. Guess-phase transients and per-iteration
    scratch (K1_alpha, J12, integral batches, ...) are excluded -- O(N) /
    O(shblk*N) on the replicated side, transient and small on the distributed side.
  * QED (--qed), external embedding (--vembedding) and constrained SCF (--cuscf)
    are modelled only when their flag is passed; ECP integrals and property
    multipoles are not modelled.
  * Integral storage: 4c-HF is integral-direct (nothing stored); in-core DF stores
    xyK at N*N*ndf; direct DF stores nothing.
  * The replicated N x N D/G pair survives the SCF loop for 4c-HF, DFT, snK and
    direct-DF, but is freed after the guess for pure DF-HF.
  * C matrices and orbital energies are rank-0 only unless ScaLAPACK is active.
  * EigenTensors::VXC_alpha / VXC_beta (the Eigen members) are dead -- never allocated.
"""

import argparse

B = 8  # bytes per double
GiB = 1024 ** 3
MiB = 1024 ** 2
TiB = 1024 ** 4


def replicated(N, s, core2e, scalapack, df, nocc, cart, dft, cuscf):
    """Bytes of replicated Eigen state, per rank, during the SCF loop, plus the
    rank-0-only extra and the post-SCF gradient term (returned separately)."""
    N2 = N * N

    t_core = s * 2 * N2 * B if core2e else 0          # D_alpha/G_alpha (+beta)
    t_cart = s * 2 * N2 * B if (dft and cart) else 0  # D_*_cart / VXC_*_cart
    t_df = (3 * N) * B if df else 0                   # dfNorm (negligible)
    t_cuscf = N2 * B if cuscf else 0                  # P_MO (Northo^2), all ranks
    per_rank_loop = t_core + t_cart + t_df + t_cuscf

    if scalapack:
        rank0_extra = s * N * B                       # eps only; C is block-cyclic
    else:
        rank0_extra = (s * N2 + N * nocc + s * N) * B  # C_alpha(+beta), C_occ, eps

    return per_rank_loop, rank0_extra


def distributed(N, s, ndiis, dft, df, direct_df, ndf, nocc, scalapack, qed,
                vembedding, cuscf):
    """Global bytes of persistent distributed (TAMM) tensors during the SCF loop."""
    N2 = N * N

    # Fock / density working set (alpha: F, D, D_diff, D_last, F_tmp, FD, FDS, ehf_tmp
    # = 8 tensors; beta adds 7).
    fam = 8 + (7 if s == 2 else 0)
    t_fam = fam * N2 * B

    # Core one-electron: H1, S1, T1, V1 all live until the end of SCF (T1/V1 are
    # reused post-loop for the kinetic / nuclear-attraction energy decomposition).
    t_core1e = 4 * N2 * B

    # Orthogonalizer + MO coefficients: X_alpha (~N^2, shared), C_alpha (s*N^2),
    # C_occ + C_occ_T (s * 2 * N*nocc).
    t_mo = (1 + s) * N2 * B + s * 2 * N * nocc * B

    # DIIS history: diis_hist (error) + fock_hist, up to ndiis tensors each, per spin.
    t_diis = s * 2 * ndiis * N2 * B

    # KS-DFT potential.
    t_vxc = s * N2 * B if dft else 0

    # Density fitting.  xyK is resident through the loop; xyZ is the transient 3c
    # build buffer (same N*N*ndf shape) that coexists with xyK during 3c-init --
    # counted in the total so it reflects the lifetime peak, not just steady state.
    t_xyk = (N2 * ndf) * B if (df and not direct_df) else 0
    t_xyz = t_xyk
    t_vm1 = (ndf * ndf) * B if df else 0

    # ScaLAPACK block-cyclic duplicates of F and C.
    t_bc = (N2 + s * N2) * B if scalapack else 0

    # Cavity-QED: 3 dipole + 6 quadrupole + QED_1body + QED_2body (+ QED_2body_beta).
    t_qed = ((11 + (1 if s == 2 else 0)) * N2 * B) if qed else 0

    # External embedding potential: Vembedding_alpha (+ beta).
    t_vemb = s * N2 * B if vembedding else 0

    # Constrained SCF: Xm1, D_cuscf, D_ortho, V_ortho, CNOS, tmp_ortho (all ~N^2),
    # plus X_comp + 2 block-cyclic tensors on the ScaLAPACK path. Spin-summed, so
    # no s factor.
    t_cuscf = ((9 if scalapack else 6) * N2 * B) if cuscf else 0

    return {
        "fam": t_fam, "core1e": t_core1e, "mo": t_mo, "diis": t_diis,
        "vxc": t_vxc, "xyK": t_xyk, "xyZ": t_xyz, "Vm1": t_vm1,
        "bc": t_bc, "qed": t_qed, "vemb": t_vemb, "cuscf": t_cuscf,
        "total": (t_fam + t_core1e + t_mo + t_diis + t_vxc + t_xyk + t_xyz
                  + t_vm1 + t_bc + t_qed + t_vemb + t_cuscf),
    }


def fmt(n):
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= TiB:
        return f"{sign}{a / TiB:.2f} TiB"
    if a >= GiB:
        return f"{sign}{a / GiB:.2f} GiB"
    return f"{sign}{a / MiB:.1f} MiB"


def main():
    p = argparse.ArgumentParser(
        description="Estimate node memory (replicated + distributed) for an ExaChem SCF run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("N", type=int, help="number of basis functions (nbf)")
    p.add_argument("-r", "--ranks-per-node", type=int, required=True, help="MPI ranks per node")
    p.add_argument("-n", "--nodes", type=int, default=1,
                   help="number of nodes in the job (distributed tensors are split over these)")
    p.add_argument("--uhf", action="store_true", help="unrestricted (UHF/UKS); default RHF/RKS")
    p.add_argument("--dft", action="store_true", help="Kohn-Sham DFT")
    p.add_argument("--scalapack", action="store_true", help="ScaLAPACK/ELPA build active")
    p.add_argument("--qed", action="store_true",
                   help="cavity QED (adds 11-12 dipole/quadrupole/self-energy N x N tensors)")
    p.add_argument("--vembedding", action="store_true",
                   help="external embedding potential (adds Vembedding_alpha (+ beta), s*N^2)")
    p.add_argument("--cuscf", action="store_true",
                   help="constrained SCF (adds ~6-9 N x N tensors distributed + P_MO replicated)")
    p.add_argument("--df", action="store_true",
                   help="3-center density fitting / RI (in-core: stores xyK at N*N*ndf)")
    p.add_argument("--direct-df", action="store_true",
                   help="with --df: recompute 3-center integrals each iteration (no xyK)")
    p.add_argument("--ndf", type=int, default=None,
                   help="auxiliary (fitting) basis size (default: 3*N)")
    p.add_argument("--cart", action="store_true",
                   help="DFT via GPU GauXC with a spherical OBS: adds Cartesian D/VXC buffers")
    p.add_argument("--nocc", type=int, default=None,
                   help="occupied orbitals (default: N // 10)")
    p.add_argument("--diis-hist", type=int, default=10,
                   help="DIIS history depth (scf_options.diis_hist)")
    p.add_argument("--gradients", action="store_true", help="also report the gradient phase")
    p.add_argument("--natoms", type=int, default=0, help="number of atoms (required with --gradients)")
    p.add_argument("--node-mem-gb", type=float, default=None,
                   help="usable RAM per node in GB; prints a PASS/FAIL verdict")
    args = p.parse_args()

    N = args.N
    R = args.ranks_per_node
    nodes = args.nodes
    s = 2 if args.uhf else 1
    nocc = args.nocc if args.nocc is not None else max(1, N // 10)
    ndf = args.ndf if args.ndf is not None else 3 * N
    if args.gradients and args.natoms <= 0:
        p.error("--gradients requires --natoms")

    core2e = (not args.df) or args.dft   # replicated D/G pair survives?

    loop, r0 = replicated(N, s, core2e, args.scalapack, args.df, nocc, args.cart,
                          args.dft, args.cuscf)
    dist = distributed(N, s, args.diis_hist, args.dft, args.df, args.direct_df,
                       ndf, nocc, args.scalapack, args.qed, args.vembedding, args.cuscf)

    nderiv = 3 * args.natoms
    t_grad = s * nderiv * N * N * B if args.gradients else 0        # replicated Ga/Gb_deriv
    dist_grad = 3 * nderiv * N * N * B if args.gradients else 0     # T_deriv + V_deriv + S_deriv

    # replicated cost of the node that hosts rank 0 (the worst node)
    rep_node_loop = R * loop + r0
    rep_node_grad = R * (loop + t_grad) + r0
    dist_per_node = dist["total"] / nodes
    dist_grad_per_node = dist_grad / nodes

    node_loop = rep_node_loop + dist_per_node
    node_grad = rep_node_grad + dist_per_node + dist_grad_per_node
    node_loop_px = rep_node_loop + 2 * dist_per_node
    node_grad_px = rep_node_grad + 2 * (dist_per_node + dist_grad_per_node)

    cfg = (f"{'UHF' if args.uhf else 'RHF'}"
           f"{'  DFT' if args.dft else ''}"
           f"{'  ScaLAPACK' if args.scalapack else ''}"
           f"{('  DF/RI(direct)' if args.direct_df else '  DF/RI(in-core)') if args.df else ''}"
           f"{'  +cart' if (args.dft and args.cart) else ''}"
           f"{'  QED' if args.qed else ''}"
           f"{'  Vembedding' if args.vembedding else ''}"
           f"{'  CUSCF' if args.cuscf else ''}")

    print()
    print(f"  N = {N}   nodes = {nodes}   ranks/node = {R}   (total ranks {nodes * R})")
    print(f"  {cfg}")
    knobs = f"  nocc = {nocc}   diis_hist = {args.diis_hist}"
    if args.df:
        knobs += f"   ndf = {ndf}"
    print(knobs)
    print(f"  replicated N x N D/G pair: {'YES' if core2e else 'no (freed after guess)'}")
    print()

    labels = {
        "fam": "Fock/density working set (F, D, D_diff, D_last, FD, FDS, ...)",
        "core1e": "core 1e integrals (H1, S1, T1, V1)",
        "mo": "orthogonalizer + MO coeffs (X, C, C_occ)",
        "diis": "DIIS history (diis_hist + fock_hist)",
        "vxc": "KS-DFT potential (VXC)",
        "xyK": "DF 3-index tensor (xyK, N*N*ndf)",
        "xyZ": "DF 3c build buffer (xyZ, N*N*ndf, transient)",
        "Vm1": "DF metric inverse (Vm1, ndf*ndf)",
        "bc": "ScaLAPACK block-cyclic copies (F_BC, C_BC)",
        "qed": "cavity-QED dipole/quadrupole/self-energy (QED_Dx..Qzz, QED_1body/2body)",
        "vemb": "external embedding potential (Vembedding_alpha (+ beta))",
        "cuscf": "constrained SCF (Xm1, D_cuscf, D_ortho, V_ortho, CNOS, tmp_ortho)",
    }
    print("  DISTRIBUTED (TAMM) tensors -- global, then per node")
    for k in ("fam", "core1e", "mo", "diis", "vxc", "xyK", "xyZ", "Vm1", "bc",
              "qed", "vemb", "cuscf"):
        if dist[k]:
            print(f"    {fmt(dist[k]):>12s}  global   {labels[k]}")
    print(f"    {fmt(dist['total']):>12s}  global   ->  {fmt(dist_per_node)} / node   TOTAL")
    print(f"    {fmt(2 * dist['total']):>12s}  global   ->  {fmt(2 * dist_per_node)} / node   "
          f"TOTAL x2 (POSIX shared-memory)")
    print()

    print("  REPLICATED (Eigen) -- per rank, then per node (node with rank 0)")
    if args.cuscf:
        print(f"    (includes P_MO {fmt(N * N * B)}/rank from CUSCF, all ranks)")
    print("    SCF loop:")
    print(f"      per rank (generic) : {fmt(loop)}")
    print(f"      per rank (rank 0)  : {fmt(loop + r0)}   (+{fmt(r0)} C/eps)")
    print(f"      per node           : {fmt(rep_node_loop)}   ({R} x rank + rank-0 extra)")
    if args.gradients:
        print(f"    gradient phase (+{fmt(t_grad)}/rank Ga/Gb_deriv, {3 * args.natoms} matrices):")
        print(f"      per rank (generic) : {fmt(loop + t_grad)}")
        print(f"      per rank (rank 0)  : {fmt(loop + t_grad + r0)}")
        print(f"      per node           : {fmt(rep_node_grad)}   "
              f"({R} x rank + rank-0 extra)")
    print()

    print("  NODE TOTAL                                                  plain     POSIX-shm")
    print(f"    SCF loop  (replicated/node + distributed/node)       : "
          f"{fmt(node_loop):>10s}  {fmt(node_loop_px):>10s}")
    if args.gradients:
        print(f"    gradient phase (+Ga/Gb_deriv, +T/V/S_deriv)          : "
              f"{fmt(node_grad):>10s}  {fmt(node_grad_px):>10s}")
        if args.df:
            print("    NOTE: DF gradient also builds 2c/3c derivative integrals "
                  "(not sized here) -- add margin.")

    if args.node_mem_gb is not None:
        budget = args.node_mem_gb * 1e9
        phases = [node_loop] + ([node_grad] if args.gradients else [])
        phases_px = [node_loop_px] + ([node_grad_px] if args.gradients else [])
        worst = max(phases)          # lifetime peak across every phase
        worst_px = max(phases_px)
        print()
        print(f"  node budget {args.node_mem_gb} GB  vs  lifetime peak:")
        print(f"    plain      {fmt(worst)}   ->  "
              f"{'PASS' if budget > worst else 'FAIL'}  (headroom {fmt(budget - worst)})")
        print(f"    POSIX shm  {fmt(worst_px)}   ->  "
              f"{'PASS' if budget > worst_px else 'FAIL'}  (headroom {fmt(budget - worst_px)})")

        print("\n  NOTE: excludes transient scratch (integral batches; guess phase;")
        print("  gradient K-vectors) and GauXC grid/collocation memory for DFT.")
    print()


if __name__ == "__main__":
    main()
