This folder bundles Planck TT source data used by `cmb.loglike`.

File:
- TT_power_spec.txt: copied from https://github.com/htjb/cmb-likelihood (data/TT_power_spec.txt)

JO runtime usage:
- Parse rows tagged as `Planck unbinned` or `Planck binned`.
- Convert D_l to C_l using C_l = D_l * 2pi / (ell * (ell + 1)).
- Use the native Planck 111-point l_center grid directly in `cmb.loglike`.
