# pyALMA4
_a Python toolkit for computing loading and tidal viscoelastic Love numbers_

This repository contains the pyALMA4 Python module and related benchmark and example Jupyter notebooks. Those are discussed in detail in [Melini and Spada (2027)](https://doi.org/10.1016/j.icarus.2026.117326).

pyALMA4 is released by Daniele Melini (daniele.melini@ingv.it) and Giorgio Spada (giorgio.spada@unibo.it) under the GNU General Public License, a copy of which is available in this repository. If you find pyALMA4 useful for your research, please cite the following paper: 
* D. Melini and G. Spada (2027), A fully compressible viscoelastic framework for modeling the planetary response to surface loading and tidal forces, Icarus, 461, 117326, https://doi.org/10.1016/j.icarus.2026.117326

pyALMA4 can be used either as a [Python library](#using-pyalma4-from-a-python-code) or as a [standalone command-line program](#using-pyalma4-from-the-command-line). Additional details are given below.

# Invoking pyALMA4 from Python

pyALMA4 can be invoked from a Python code or a Jupyter notebook as follows:

``` 
import alma4
output = alma4.love_numbers(r,rho,mu,lam,eta,rheology,params,    \
                           degrees,timesteps,loadtype,analysis, \
                           verbose=False, timeunits='kyr', order=8, \
                           xi=1e-4, n0=20, numint=None, \
                           adaptive=Trueinversion='default')
```

where:
* `r, rho, mu, lam, eta` are vectors containing the outer radius (in m), density (in kg/m^3), shear rigidity and 1st Lamé parameter (in Pa) and viscosity (in Pa*s) for each layer in the model. Layers shall be ordered from the innermost to the outermost one (i.e., `r[0]` is the core radius and `r[-1]` is the external surface radius). Set `lam=None` for an incompressible model. For elastic and fluid layers the value of `eta` is ignored. All vectors must have the same length. Note that for some rheologies, viscosity and/or Lamé parameters may be ignored ([see the table below](#available-rheological-models)).
* `rheology`, `params` are vectors containing the rheology of each layer and the corrsponding rheological parameters, where needed. `params` must be an array with a number of rows equal to the number of layers and at least two columns, with `params[i,j]` containing parameter $p_j$ for the $i$-th layer ([see the table below](#available-rheological-models) for the definition of required parameters for each of the implemented rheological laws).
* `degrees`: scalar or vector with harmonic degree(s) for which the LNs will be computed.
* `timesteps`: scalar or vector with timestep(s) at which Heaviside LNs are evaluated (for `heaviside` analysis, see below), or period(s) at which LNs for a periodic forcing are evaluated (`frequency` analysis), or inverses of the sampling points on the real positive axis (`laplace` or `collocation` analysis). For other analyses it is ignored. Time steps (or forcing periods) can be given in units of `kyr`, `yr`, `day` or `hr` by setting the `timeunits` parameter (see below). Default units are `kyr`. See the [table below](#available-analyses) for a summary of the meaning of  `timesteps` in different analyses.
* `loadtype`: can be `'loading'` for loading LNs or `'tidal'` for tidal LNs. Loading LNs can be computed for degree $n\ge1$ while tidal LNs for degree $n\ge 2$. 
* `analysis`: analysis type. See  the [table below](#available-analyses) for available options.
* `verbose`: if `True`, print progress and timing info (default is `False`).
* `timeunits`: units for time steps (`timesteps`) and for rheological parameters, where applicable. Valid units are `kyr`, `yr`, `day`, `hr`. Default is `kyr`.  
* `order`: order of the Talbot or Post-Widder inversions used to retrieve Heaviside LNs. Default is `order=8`.
* `xi,n0`: Parameters defining at which radius the outward propagation of the solution will start for harmonic degree `n>n0`. See [Melini and Spada (2027)](https://doi.org/10.1016/j.icarus.2026.117326) for details.  
* `numint`: If set to `True`, numerical Runge-Kutta integration is used also for incompressible models, while if set to `False` analytical propagation is used also for compressible models. Default (`numint=None`) is to use Runge-Kutta integration for compressible models and analytical propagation for incompressible models.
* `adaptive`: if set to `True` (default), the adaptive Post-Widder inversion scheme introduced by [Caron et al. (2026)](https://doi.org/10.5194/gmd-19-4031-2026) will be adopted.
* `inversion`: Laplace inversion scheme used for retrieving time-domain Heaviside LNs. Set to `postwidder` to use the Post-Widder formula; the default is the Fixed-Talbot algorithm.

On output, pyALMA4 returns:
* for the `elastic`, `fluid` or `fluidlimit` analyses: `output = (h, l, k)` with `h, l, k` being 1D arrays of length `len(degrees)`.
* for the `heaviside`, `frequency` or `laplace` analyses: `output = (h, l, k)` with `h, l, k` being 2D arrays of size `(len(degrees), len(timesteps))`. For the `frequency` analysis, those arrays are complex-valued.
* for the `collocation` analysis: `output = (he, le, ke, hv, lv, kv)`, with `he, le, ke` being 1D arrays of length `len(degrees)` containing the elastic LNs and 
`hv, lv, kv` being 2D arrays of size `(len(degrees), len(timesteps))` that define the viscoelatic response of the model ([see below](#note-about-the-collocation-analysis) for further details).



# Using pyALMA4 from the command line

pyALMA4 can be invoked from the shell prompt as follows:

```
$ run-alma4.py  --degrees=DEGREES \
                --model=MODEL
                --analysis=ANALYSIS
                [--incompressible] \
                [--log] \
                (--loading | --tidal) \
                [--timesteps=TIMESTEPS] \ 
                [--timescale={log,linear}] \
                [--label=LABEL] \
                [--timeunits={kyr,yr,days,hr}] \
                [--output-format={ln_vs_n,ln_vs_t}] \
                [--propagation={analytical,numerical,default}] \ 
                [--inversion={postwidder,talbot,default}]
```

where
* `--analysis=ANALYSIS` (required): analysis type. See [below](#available-analyses) for a list of available analyses.
* `--model=MODEL` (required): full path to the rheological model file. The model file shall contain one row per layer (ordered from the outermost to the innermost one), with header lines starting with `#` being interpreted as comments and ignored. Each line is expected to contain the following columns (see also the `MODELS` directory for some examples of model files):
    * 1st column: layer number (present only for the user's convenience and ignored by pyALMA4)
    * 2nd column: outer radius (in m)
    * 3rd column: density (in kg/m^3)
    * 4th and 5th columns: Lamé parameters $\mu$ and $\lambda$ (in Pa)
    * 6th column: viscosity $\eta$ (in Pa*s)
    * 7th column: rheology ([see below](#available-rheological-models) for a list of available rheological models)
    * 8th-11th columns: additional rheological parametes, if needed ([see below](#available-rheological-models) for their definition)
* `--degrees=DEGREES` (required): range of harmonic degrees. Specify a comma-separated list of blocks of the form `nmin[:nmax[:step]]`. For example:
    * `--degrees=2,5,10` corresponds to $n=2,5,10$
    * `--degrees=1:5,10,20` corresponds to $n=1,2,3,4,5,10,20$ 
    * `--degrees=1:5,10:20:5,50` corresponds to $n=1,2,3,4,5,10,15,20,50$.
* `--incompressible`: assume an incompressible model (values of the Lamé parameter $\lambda$ in the model file will be ignored).
* `--loading` or `--tidal`: select loading or tidal LNs. One of the two options must be selected.
* `--timesteps=TIMESTEPS`: specifies the time steps. Give a comma-separated list of times or of blocks of the form `tmin:tmax:delta_t`. Note that all time steps shall be $>0$. If `delta_t` is a negative number, `-delta_t` equally spaced samples between `tmin` and `tmax` will be generated. For example:
   * `--timesteps=10,15,50`: selects timesteps $t=10,15,50$ kyr.
   * `--timesteps=1,2,5,10:100:10`: selects timesteps $t=1,2,5,10,20,\ldots,100$ kyr.
   * `--timesteps=1,2,4,10:100:-4,200`: selects timesteps $t=1,2,5,10,40,70,100,200$ kyr.
   * `--timesteps=-4:4:1` (with the `--timescale=log` option, see below): selects timesteps $t=10^-4, 10^-3, \ldots, 10^3, 10^4$ kyr.
* `--timescale=TIMESCALE`: selects if the timesteps are defined on a linear scale (`linear`) or on a log scale (`log`). In the latter case, time steps are defined as 10 to the power of the values specified with the `--timesteps` option.
* `--timeunits=TIMEUNITS`: selects the units in which the time steps are given. Valid options are `kyr` (default), `yr`, `day`, `hr`.
* `--log`: write a log file `alma.log` (or `alma-LABEL.log`, if the `--label` option is set) with additional information about the run. Default is to not write a log file.
* `--label=LABEL`: define a label that will be added to the output file names and to the log file name, if requested (see the `--log` option). If the label is defined, output file names will be `h-LABEL.dat`, `l-LABEL.dat` and `k-LABEL.dat`; otherwise they will be `h.dat`, `l.dat`, `k.dat`. Note that for the `frequency` analysis, two files for each Love number will be created, containing respectively the real and imaginary parts and identified by rhe `re_` and `im_` prefixes.
* `--output-format=FORMAT`: selects the format of the output files. Valid options are `ln_vs_n` (each row corresponds to an harmonic degree and colums correspond to timesteps) and `ln_vs_t` (each row correspons to a timestep and columns correspond to harmonic degrees). For `elastic`, `fluid`, `fluidlimit` and `collocation` analysis the option is ignored and output files are always created in the `ln_vs_n` format.
* `--propagation=PROPAGATION`: selects the method used to propagate the solution in solid layers. Valid options are `analytical` or `numerical`. Default is `analytical` for incompressible models and `numerical` for compressible models.
* `--inversion=INVERSION`: selects the Laplace inversion method for the `heaviside` analysis. Valid options are `talbot` (default) or `postwidder`.

On exit, pyALMA4 will create output files named `h.dat`, `l.dat` and`k.dat` (or `h-LABEL.dat`, `l-LABEL.dat`, `k-LABEL.dat` if the `--label` option is set), containing results for $h$, $l$ and $k$ Love numbers. For the `frequency` analysis, two files for each LN will be created, containing the real and imaginary parts of the LN and identified by the `re_` and `im_` prefixes, respectively. The contents of the output files is organized according to the format selected with the `--output-format` option.

# Available analyses

| Analysis | Output | Timesteps |
| -------- | ----------- | --------- |
`elastic` | Elastic Love numbers<br>(an elastic rheology is assumed in all solid layers) | ignored 
`fluid`   | Fluid Love numbers<br>(an inviscid fluid rheology assumed in all layers) | ignored |
`fluidlimit` | Fluid limit of viscoelastic Love numbers<br>(an inviscid fluid rheology assumed is in all layers except those with `elastic` rheology) | ignored |
`laplace` | Laplace-transformed Love numbers | Inverse of sampling points on the real positive axis, $t_i=1/s_i$ |
`heaviside` | Time-dependent viscoelastic Love numbers for an Heaviside forcing $H(t)$ | Time steps at which the LNs are evaluated |
`timedomain` | Synonym of `heaviside` | See above |
`frequency` | Complex Love numbers for a periodic load | Periods $T_i$ of the forcing |
`periodic` | Synonym of `frequency` | See above |
`collocation` | Collocation analysis of Laplace-transformed Love numbers<br>(see [below](#note-about-the-collocation-analysis) for more details) | Inverse of sampling points on the real positive axis, $t_i=1/s_i$
| | |


# Available rheological models

| Rheology | Description | Additional parameters | Notes |
| -------- | ----------- | -------------| ----- | 
| `elastic` | Elastic | - | Viscosity $\eta$ is ignored |
| `fluid`   | Inviscid fluid | - | Viscosity $\eta$ and Lamé parameters $\lambda$, $\mu$ are ignored | 
| `newton`   | Newtonian fluid | - |Lamé parameters $\lambda$, $\mu$ are ignored | 
| `maxwell`  | Maxwell | - | - |
| `burgers`  | Burgers | $p_0=\mu_K/\mu_M$ <br> $p_1=\eta_K/\eta_M$ | $\mu_K$, $\eta_K$: shear rigidity and viscosity of the Kelvin-Voigt element <br>$\mu_M$, $\eta_M$: shear rigidity and viscosity of the Maxwell element |
| `andrade`  | Andrade | $p_0=\alpha$ <br> $p_1=\zeta$ | $\alpha$:  fractional creep exponent ($0<\alpha<1$) <br> $\zeta$: ratio between the transient relaxation time $\tau_A$ and the steady-state relaxation time $\tau_M$ of the Maxwell element. |
| `sundberg` | Sundberg-Cooper | $p_0=\mu_K/\mu_M$ <br> $p_1=\eta_K/\eta_M$ <br> $p_2=\alpha$ <br> $p_3=\zeta$ | See notes to the Burgers and Andrade rheologies above for the physical meaning of parameters. |
| `ebm`      | Extended Burgers |  $p_0=\alpha$ <br> $p_1=\Delta$ <br> $p_2=\tau_L$ <br> $p_3=\tau_H$ | $\alpha$:  exponent of the Boltzmann distribution of relaxation times ($-1<\alpha<1$) <br> $\Delta$: ratio between unrelaxed and relaxed moduli <br> $\tau_L$, $\tau_H$: low and high cutoffs in the distribution of relaxation times. Note that $\tau_L$, $\tau_H$ shall be expressed in the same units of the timesteps |
| | | |

# Note about the `collocation` analysis

The `collocation` analysis approximates the Laplace-transformed Love numbers for an impulsive forcing through the _pure collocation_ method. If $x$ denotes one of the $h$, $l$ or $k$ Love numbers, its Laplace transform $\tilde{x}$ is approximated as

$$
\tilde{x}_n(s)=x_{e,n} + \sum_{i=1}^{N_s} \frac{x_{n,i}}{s+s_i}
$$

where the $N_s$ sampling points $s_i$ on the real positive axis are provided by the user. 

The form above of the Laplace-transformed LNs is identical to the one obtained in the context of viscoelastic normal modes (see, e.g., [Spada et al., 2001](https://doi.org/10.1111/j.1365-246X.2011.04952.x)) and can be used to compute analytically the time-domain LNs for a wide range of load forcing time-histories.  In a `collocation` analysis, pyALMA4 evaluates the elastic limit $x_{e,n}$ and the $N_s$ weights $x_{n,i}$ for each Love number. For further details about the collocation method, see [Mitrovica and Peltier, 1992](https://doi.org/10.1111/j.1365-246X.1992.tb04623.x).
