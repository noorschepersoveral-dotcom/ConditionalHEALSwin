import numpy as np
import healpy as hp

def make_phase_locked_grf(
    m_in,
    lmax=None,
    cl_target=None,
    iter=3,
    exact_cl=True,
    use_pixel_weights=False,
    seed=None,
):
    """
    Build a phase-locked, synfast-like scalar HEALPix map.

    Parameters
    ----------
    m_in : array-like, shape (npix,)
        Input scalar HEALPix map.
    lmax : int or None
        Maximum multipole. If None, use 3*nside - 1.
    cl_target : array-like or None
        Target C_ell. If None, use the auto-spectrum of m_in.
    iter : int
        Number of map2alm iterations.
    exact_cl : bool
        If True, renormalize each ell-shell so that the output C_ell matches
        cl_target exactly in harmonic space.
        If False, keep the raw synalm amplitudes, so the spectrum fluctuates
        around cl_target like an ordinary synfast realization.
    use_pixel_weights : bool
        Passed to hp.map2alm.
    seed : int or None
        Random seed for the synalm draw.

    Returns
    -------
    m_out : ndarray
        Output scalar HEALPix map.
    """
    m_in = np.asanyarray(m_in)
    if m_in.ndim != 1:
        raise ValueError(
            "This function expects a single scalar HEALPix map with shape (npix,)."
        )

    nside = hp.get_nside(m_in)
    if lmax is None:
        lmax = 3 * nside - 1

    alm_in = hp.map2alm(
        m_in,
        lmax=lmax,
        iter=iter,
        pol=False,
        use_pixel_weights=use_pixel_weights,
    )

    if cl_target is None:
        cl_target = hp.alm2cl(alm_in, lmax=lmax)
    else:
        cl_target = np.asarray(cl_target, dtype=float)
        if cl_target.ndim != 1 or cl_target.size < lmax + 1:
            raise ValueError("cl_target must be 1D and have length >= lmax + 1.")
        cl_target = cl_target[: lmax + 1]

    if np.any(cl_target < 0):
        raise ValueError("cl_target must be non-negative for a scalar auto-spectrum.")

    # Draw Gaussian amplitudes with the target spectrum.
    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)
    try:
        alm_g = hp.synalm(cl_target, lmax=lmax)
    finally:
        if seed is not None:
            np.random.set_state(rng_state)
          
    amp_in = np.abs(alm_in)
    phase_in = np.ones_like(alm_in, dtype=np.complex128)
    nonzero = amp_in > 0
    phase_in[nonzero] = alm_in[nonzero] / amp_in[nonzero]

    # Replace random phases by the input phases.
    alm_out = np.abs(alm_g) * phase_in

    # For a real map, m=0 coefficients must be purely real.
    _, m = hp.Alm.getlm(lmax)
    alm_out[m == 0] = alm_out[m == 0].real

    if exact_cl:
        # Renormalize each ell shell so C_ell matches cl_target exactly.
        cl_now = hp.alm2cl(alm_out, lmax=lmax)

        fl = np.ones(lmax + 1, dtype=float)
        good = cl_now > 0
        fl[good] = np.sqrt(cl_target[good] / cl_now[good])
        fl[(~good) & (cl_target == 0)] = 0.0

        bad = (~good) & (cl_target > 0)
        if np.any(bad):
            bad_ell = np.where(bad)[0]
            raise RuntimeError(f"Zero shell power encountered at ell={bad_ell}.")

        alm_out = hp.almxfl(alm_out, fl)

    m_out = hp.alm2map(alm_out, nside, lmax=lmax, pol=False)
    return m_out

N_cosmo = 3
N_real  = 500
nside   = 64
npix    = hp.nside2npix(nside)

maps = Target

ini_de_cosmo0 = np.zeros((N_real, npix), dtype=np.float64)
ini_de_cosmo1 = np.zeros((N_real, npix), dtype=np.float64)
ini_de_cosmo2 = np.zeros((N_real, npix), dtype=np.float64)

output_containers = ini_de_cosmo0, ini_de_cosmo1, ini_de_cosmo2

for cosmo_idx in range(N_cosmo):
    template_map = maps[cosmo_idx]          
    mu_target    = template_map.mean()
    sig_target   = template_map.std()

    print(f"\nCosmology {cosmo_idx:04d}  |  mean={mu_target:.6f}  std={sig_target:.6f}")

    for real_idx in range(N_real):
        seed = cosmo_idx * N_real + real_idx

        m_out = make_phase_locked_grf(
            m_in          = template_map,
            seed          = seed,
        )

        output_containers[cosmo_idx][real_idx] = m_out

        print(f"  real {real_idx:02d}  mean={m_out.mean():.6f}  std={m_out.std():.6f}")

# --- Sanity check ---
print("\n--- Summary ---")
for cosmo_idx, (container, name) in enumerate(zip(
    output_containers,
    ["ini_de_cosmo0", "ini_de_cosmo1", "ini_de_cosmo2"]
)):
    template_map = maps[cosmo_idx]
    print(
        f"{name} | "
        f"target mean={template_map.mean():.6f}  std={template_map.std():.6f} | "
        f"output mean={container.mean(axis=1).mean():.6f}  "
        f"std={container.std(axis=1).mean():.6f}"
    )
