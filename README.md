# pyALMA4
a Python toolkit for computing loading and tidal viscoelastic Love numbers

This repository contains the `alma4` Python module and related benchmark and example Jupyter notebooks. The code is described in Melini and Spada (2026), submitted to Computer and Geosciences.

The syntax for computing viscoeastic Love Numbers with pyALMA4 is described below:

``` 
h,l,k = alma4.love_numbers(r,rho,mu,lam,eta,rheology,params,    \
                           degrees,timesteps,loadtype,analysis, \
                        verbose=False, order=8, xi=1e-4, n0=20, \
                        numint=None,adaptive=True,inversion='default')
```

* `r, rho, mu, lam, eta`: vectors containing the outer radius (in m), density (in kg/m^3), shear rigidity and 1st Lamé parameter (in Pa) and viscosity (in Pa) for each layer in the model. Layers shall be ordered from the innermost to the outermost one (i.e., `r[0]` is the core radius and `r[-1]` is the surface radius). Set `lam=None` for an incompressible model. For elastic and fluid layers the value of `eta` is ignored. All vectors must have the same length.
* `rheology`, `params`: vectors containing the rheology of each layer and the corrsponding rheological parameters. `params` must be an array with a number of rows equal to the number of layers and at least two columns. Additional columns may be required according to the rheological laws defined in the layers (see below for more details). Supported rheological laws and corresponding expected values for `params` are listed below. 
    * `elastic`: elastic rheology. Values of `eta` and `params` are ignored.
    * `fluid`: inviscid fluid rheology. Values of `mu`, `lam`, `eta`, and `params` are ignored.
    * `newton`: Newton rheology. Values of `mu`, `lam`, and `params` are ignored.
    * `kelvin`: Kelvin-Voigt rheology. Values of `params` are ignored.
    * `maxwell`: Maxwell rheology. Values of `params` are ignored.
    * `burgers`: Burgers transient rheology. Set `mu`$=\mu_M$, `eta`=$\eta_M$, `params[:,0]`=$\mu_K/\mu_M$, `params[:,1]`=$\eta_K/\eta_M$, with subscripts $M$ and $K$ referring to the Maxwell and Kelvin-Voigt elements of the mechanical analogue for the Burgers solid.
    * `andrade`: Andrade transient rheology. Set `params[:,0]` to the Andrade exponent $\alpha$ and `params[:,1]` to the ratio $\zeta = \tau_A/\tau_M$ between the Andrade and Maxwell relaxation times.
    * `sundberg`: Sundberg-Cooper transient rheology. Set `mu`$=\mu_M$, `eta`=$\eta_M$, `params[:,0]`=$\mu_K/\mu_M$, `params[:,1]`=$\eta_K/\eta_M$, `params[:,2]`=$\alpha$ (the fractional creep exponent) and `params[:,3]`=$\zeta$ (the ratio between the transient and steady-state relaxation times).
    * `ebm`: Extended Burgers rheology. Set `params[:,0]`=$\alpha$ (the exponent of the Boltzmann distribution of relaxation times), `params[:,1]`=$\Delta$ (the ratio between unrelaxed and relaxed moduli), `params[:,2]`=$\tau_L$ and `params[:,3]`=$\tau_H$ (the low and high cutoffs in the distribution of relaxation times, expressed in kyr).
* `degrees`: scalar or vector with harmonic degree(s) for which the LNs will be computed.
* `timesteps`: scalar or vector with timestep(s) at which Heaviside LNs are evaluated (for `heaviside` analysis, see below) or period(s) at which LNs for a periodic forcing are evaluated (`frequency` analysys), see below. For other analyses it is ignored. Time steps of periods must be given in units of kyr.
* `loadtype`: can be `'loading'` for loading LNs or `'tidal'` for tidal LNs. Loading LNs can be computed for degree $n\ge1$ while tidal LNs for degree $n\ge 2$. 
* `analysis`: can be one of the following
    * `elastic`: compute the elastic limit of the LNs (all solid layers are assumed to be elastic).
    * `fluid`: compute the fluid limit of the LNs (all viscoelastic layers are assumed to be fluid).
    * `heaviside`: Compute the viscoelastic real LNs for a Heaviside forcing time history, at the timesteps specified by the `timesteps` parameter.
    * `frequency`: Viscoelastic complex LNs for a periodic forcing time history, at the periods specified by the `timesteps` parameter.
* `verbose`: if `True`, print progress and timing info (default is `False`).
* `order`: order of the Talbot or Post-Widder inversions used to retrieve Heaviside LNs. Default is `order=8`.
* `xi,n0`: Parameters defining at which radius the outward propagation of the solution will start for harmonic degree `n>n0`. See Melini and Spada (2026) for details.  
* `numint`: If set to `True`, numerical Runge-Kutta integration is used also for incompressible models, while if set to `False` analytical propagation is used also for compressible models. Default (`numint=None`) is to use Runge-Kutta integration for compressible models and analytical propagation for incompressible models.
* `adaptive`: if set to `True` (default), the adaptive Post-Widder inversion scheme introduced by Caron et al. (2025) will be adopted.
* `inversion`: Laplace inversion scheme used for retrieving time-domain Heaviside LNs. Set to `postwidder` to use the Post-Widder formula; the default is the Fixed-Talbot algorithm.

On output, the `h, l, k` arrays are returned, with size `(len(degrees), len(timesteps))`. If `analysis='frequency'` the arrays are complex, otherwise they are real-valued.

