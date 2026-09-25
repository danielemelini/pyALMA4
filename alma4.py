#
#                                                      ___
#                                                   ,o88888
#                                                ,o8888888'
#                          ,:o:o:oooo.        ,8O88Pd8888"
#                      ,.::.::o:ooooOoOoO. ,oO8O8Pd888'"
#                    ,.:.::o:ooOoOoOO8O8OOo.8OOPd8O8O"
#                   , ..:.::o:ooOoOOOO8OOOOo.FdO8O8"
#                  , ..:.::o:ooOoOO8O888O8O,COCOO"
#                 , . ..:.::o:ooOoOOOO8OOOOCOCO"
#                  . ..:.::o:ooOoOoOO8O8OCCCC"o
#                     . ..:.::o:ooooOoCoCCC"o:o
#                     . ..:.::o:o:,cooooCo"oo:o:
#                  `   . . ..:.:cocoooo"'o:o:::'
#                  .`   . ..::ccccoc"'o:o:o:::'
#                 :.:.    ,c:cccc"':.:.:.:.:.'
#               ..:.:"'`::::c:"'..:.:.:.:.:.'
#             ...:.'.:.::::"'    . . . . .'
#            .. . ....:."' `   .  . . ''
#          . . . ...."'
#          .. . ."'
#         .
#
#
# pyALMA, version 4.01
# (the plAnetary Love nuMbers cAlculator)

import math
import time
import numpy as np
from scipy.special import binom
from scipy.special import gamma
from scipy.special import spherical_jn, spherical_yn    
from scipy.special import hyp2f1
from collections import namedtuple
from numpy.lib.scimath import sqrt
from scipy.integrate import odeint, solve_ivp

pi = np.pi

def direct_matrix(n,r,rho,mu,gra,G):
    """
    Computes the incompressible fundamental matrix.
    """

    a1 = 2*n + 3             # 2n+3
    a2 = n + 1               # n+1
    a3 = n + 3               # n+3
    a4 = n**2 - n - 3        # n**2-n-3
    a5 = n + 2               # n+2
    a6 = n - 1               # n-1
    a7 = 2 * n + 1           # 2n+1
    a8 = 2 * n - 1           # 2n-1
    a9 = 2 - n               # 2-n
    a10= n**2 + 3 * n - 1    # n**2+3n-1
    a11= n**2 - 1            # n**2-1

    Y = np.zeros((6,6),np.complex128)
 
    Y[0,0] = n/(2 * a1) * r**(n+1)
    Y[1,0] = a3 / ( 2 * a1 * a2 ) * r**(n+1) 
    Y[2,0] = ( n * rho * gra * r + 2 * a4 * mu ) / ( 2 * a1 ) * r**n
    Y[3,0] = n * a5 / ( a1 * a2 ) * mu * r**n
    Y[5,0] = 2 * pi * G * rho * n / a1 * r**(n+1)

    Y[0,1] = r**(n-1)
    Y[1,1] = 1 / n * r**(n-1)
    Y[2,1] = ( rho * gra * r + 2 * a6 * mu ) * r**(n-2)
    Y[3,1] = 2 * a6 / n * mu * r**(n-2)
    Y[5,1] = 4 * pi * G * rho * r**(n-1)

    Y[2,2] = rho * r**n
    Y[4,2] = r**n
    Y[5,2] = a7 * r**(n-1)

    Y[0,3] = a2 / ( 2 * a8 ) * r**(-n)
    Y[1,3] = a9 / ( 2 * n * a8 ) * r**(-n)
    Y[2,3] = ( a2 * rho * gra * r - 2 * a10 * mu ) / ( 2 * a8 ) * r**(-n-1)
    Y[3,3] = a11 / ( n * a8 ) * mu * r**(-n-1)
    Y[5,3] = 2 * pi * G * rho * a2 / a8 * r**(-n)

    Y[0,4] = r**(-n-2)
    Y[1,4] = - 1.0 / a2 * r**(-n-2)
    Y[2,4] = ( rho * gra * r - 2 * a5 * mu ) * r**(-n-3)
    Y[3,4] = 2 * a5 / a2 * mu * r**(-n-3)
    Y[5,4] = 4 * pi * G * rho * r**(-n-2)
    
    Y[2,5] = rho * r**(-n-1)
    Y[4,5] = r**(-n-1)

    return Y



def inverse_matrix(n,r,rho,mu,gra,G):
    """
    Computes the inverse of the incompressible fundamental matrix.
    """

    a1 = 2 * n + 1           # 2n+1
    a2 = 2 * n - 1           # 2n-1
    a3 = n + 1               # n+1
    a4 = n - 1               # n-1
    a5 = n + 2               # n+2
    a6 = n + 3               # n+3
    a7 = n**2 + 3 * n - 1    # n**2+3n-1
    a8 = n**2 - n - 3        # n**2-n-3
    a9 = 2 - n               # 2-n
    a10= n**2 - 1            # n**2-1
    a11= 2 * n + 3           # 2n+3

    Yinv = np.zeros((6,6),np.complex128)
    D    = np.zeros((6,6),np.complex128)

    D[0,0] = a3 * r**(-(n+1))
    D[1,1] = a3 * n / ( 2 * a2 ) * r**(-(n-1))
    D[2,2] = -r**(-(n-1))
    D[3,3] = n * r**n
    D[4,4] = n * a3 / ( 2 * a11 ) * r**(n+2)
    D[5,5] = r**(n+1)

    D = D / a1
 
    Yinv[0,0] =   rho * gra * r / mu - 2 * a5
    Yinv[1,0] = - rho * gra * r / mu + 2 * a7 / a3
    Yinv[2,0] =   4 * pi * G * rho
    Yinv[3,0] =   rho * gra * r / mu + 2 * a4
    Yinv[4,0] = - rho * gra * r / mu - 2 * a8 / n
    Yinv[5,0] =   4 * pi * G * rho * r

    Yinv[0,1] =   2 * n * a5
    Yinv[1,1] = - 2 * a10
    Yinv[3,1] =   2 * a10
    Yinv[4,1] = - 2 * n * a5

    Yinv[0,2] = - r / mu
    Yinv[1,2] =   r / mu
    Yinv[3,2] = - r / mu
    Yinv[4,2] =   r / mu

    Yinv[0,3] =   n * r / mu
    Yinv[1,3] =   a9 * r / mu
    Yinv[3,3] = - a3 * r / mu
    Yinv[4,3] =   a6 * r / mu
    
    Yinv[0,4] =   rho * r / mu
    Yinv[1,4] = - rho * r / mu
    Yinv[3,4] =   rho * r / mu
    Yinv[4,4] = - rho * r / mu
    Yinv[5,4] =   a1

    Yinv[2,5] = - 1
    Yinv[5,5] = - r

    Yinv = np.matmul(D, Yinv)

    return Yinv


def direct_fluid_matrix(n,r,rho,gra,G):
    """
    Computes the incompressible fluid fundamental matrix.
    """

    Y = np.zeros( (2,2), dtype=np.float64 )

    Y[0,0] = r**n
    Y[0,1] = r**(-n-1)

    Y[1,0] = (2*n+1) * r**(n-1) - 4 * pi * G * rho/gra * r**n
    Y[1,1] = -4 * pi * G * rho/gra * r**(-n-1)

    return Y

def inverse_fluid_matrix(n,r,rho,gra,G):
    """
    Computes the inverse of the incompressible fluid fundamental matrix.
    """

    Yinv = np.zeros( (2,2), dtype=np.float64 )

    Yinv[0,0] = 4 * pi * G  *rho/gra * r**(-n+1)
    Yinv[0,1] = r**(-n+1)

    Yinv[1,0] = ((2*n+1) - 4 * pi * G * rho/gra * r) * r**(n+1)
    Yinv[1,1] = -r**(n+2)

    Yinv = Yinv / (2*n+1)

    return Yinv

def surface_bc(n,r,gra,iload,G):
    """
    Computes the boundary conditions at the surface.
    """

    bs = np.zeros(3, np.float64)
    kappa = ( 2 * n + 1) / ( 4 * pi * r**2 ) 
   
    if (n!=1):
        if(iload==0):   # Tidal conditions
            bs[2] = -4 * pi * G * kappa
        else:           # Loading conditions
            bs[0] = - gra * kappa
            bs[2] = -4 * pi * G * kappa
    else:
        if(iload==0):   # Tidal conditions
            raise ValueError( "Error: tidal conditions are not defined for n=1")
        else:           # Loading conditions
            bs[0] = - gra * kappa
    
    return bs

def salzer_weights(order):
    """
    Computes the Salzer weights for the Post-Widder inversion algorithm.
    """

    if( type(order) != int ):
        raise TypeError("salzer_weights(): order must be integer")

    m = order
    zeta = np.zeros( 2*m )

    for k in range(1,2*m+1):
        j1 = math.floor( (k+1)/2 )
        j2 = min( k, m )
        for j in range(j1,j2+1):
            fattm = math.factorial(m)
            q1 = binom(m, j)
            q2 = binom(2*j, j)
            q3 = binom(j, k-j)
            zeta[k-1] = zeta[k-1] + j**(m+1) / fattm * q1 * q2 * q3
        if (m+k)%2 != 0:
            zeta[k-1] = -zeta[k-1]

    return zeta

def postwidder_fcn_inversion(fs,t,order,zeta):
    """
    Computes the Laplace inverse of a generic function through the PW algorithm
    """

    ft = 0.0

    f = math.log(2.0)/t

    for k in range(2*order):

        s = f * (k+1)      # k goes from 0 to 2*order-1

        ft += fs(s) * zeta[k] * f

    return ft

def adaptive_postwidder_fcn_inversion(fs,t,order,zeta):
    """
    Computes the Laplace inverse of a generic function through adaptive Post-Widder Laplace inversion
    """

    ft = np.zeros(order, dtype=complex)
    
    mconv = -1

    for m in range(order):
        
        ft[m] = postwidder_fcn_inversion(fs,t,m+1,zeta[m])
        
        if m>2:
            if mconv<0:
                if ( abs(ft[m]-ft[m-1]) > abs(ft[m-1]-ft[m-2]) ) and \
                      ( abs(ft[m-1]-ft[m-2]) > abs(ft[m-2]-ft[m-3]) ):
                    mconv = m-2
                    break

    return ft[mconv]

def talbot_fcn_inversion(fs,t,M):
    """
    Computes the Laplace inverse of a generic function through the Talbot method
    """
    r = 0.4*M/t

    s = r
    ft = 0.5 * fs(s) * np.exp(r*t) / s

    for k in range(1,M):
        theta = k * np.pi/M
        s   = r * theta * (1/np.tan(theta) + 1j)
        sig = theta + (theta / np.tan(theta) - 1)/np.tan(theta)
        ft += np.real( fs(s) * np.exp(t*s) * ( 1 + 1j * sig ) )

    ft = ft * 0.4/t

    return ft

def postwidder_inversion(n,t,iload,model,order,zeta,xi,n0,numint):
    """
    Computes the time-dependent Heaviside LNs through Post-Widder Laplace inversion
    """
    
    h_heav, l_heav, k_heav = 0.0, 0.0, 0.0

    f = math.log(2.0) / t

    for k in range(2*order):
            
        s = f * (k+1)      # k goes from 0 to 2*order-1
                    
        hh, ll, kk = love_numbers_spectrum(n,s,iload,model, xi=xi, n0=n0, numint=numint)
                    
        h_heav += hh * zeta[k] * f / s
        l_heav += ll * zeta[k] * f / s
        k_heav += kk * zeta[k] * f / s

    return h_heav, l_heav, k_heav

def adaptive_postwidder_inversion(n,t,iload,model,order,zeta,xi,n0,numint):
    """
    Computes the time-dependent Heaviside LNs through adaptive Post-Widder Laplace inversion
    """

    hh = np.zeros(order, dtype=complex)
    ll = np.zeros(order, dtype=complex)
    kk = np.zeros(order, dtype=complex)

    mh, ml, mk = -1, -1, -1

    for m in range(order):
        hh[m],ll[m],kk[m] = postwidder_inversion(n,t,iload,model,m+1,zeta[m],xi,n0,numint)

        if m>2:
            if mh<0:
                if ( abs(hh[m]-hh[m-1]) > abs(hh[m-1]-hh[m-2]) ) and \
                      ( abs(hh[m-1]-hh[m-2]) > abs(hh[m-2]-hh[m-3]) ):
                    mh = m-2
            if ml<0:
                if ( abs(ll[m]-ll[m-1]) > abs(ll[m-1]-ll[m-2]) ) and \
                      ( abs(ll[m-1]-ll[m-2]) > abs(ll[m-2]-ll[m-3]) ):
                    ml = m-2
            if mk<0:
                if ( abs(kk[m]-kk[m-1]) > abs(kk[m-1]-kk[m-2]) ) and \
                      ( abs(kk[m-1]-kk[m-2]) > abs(kk[m-2]-kk[m-3]) ):
                    mk = m-2
        
        if (mh>=0) and (ml>=0) and (mk>=0):
            break
    if mh<0:
        mh = order-1
    if ml<0:
        ml = order-1
    if mk<0:
        mk = order-1

    return hh[mh], ll[ml], kk[mk]

def talbot_inversion(n,t,iload,model,M,xi,n0,numint):

    r = 0.4*M/t

    s = r
    hh, ll, kk = love_numbers_spectrum(n,s,iload,model,xi,n0,numint)
    hh = 0.5 * np.real(hh) * np.exp(r*t) / s
    ll = 0.5 * np.real(ll) * np.exp(r*t) / s
    kk = 0.5 * np.real(kk) * np.exp(r*t) / s

    for k in range(1,M):
        theta = k * np.pi/M
        s   = r * theta * (1/np.tan(theta) + 1j)
        sig = theta + (theta / np.tan(theta) - 1)/np.tan(theta)
        hh1, ll1, kk1 = love_numbers_spectrum(n,s,iload,model,xi,n0,numint)
        hh += np.real( hh1 * np.exp(t*s) * ( 1 + 1j * sig ) / s )
        ll += np.real( ll1 * np.exp(t*s) * ( 1 + 1j * sig ) / s )
        kk += np.real( kk1 * np.exp(t*s) * ( 1 + 1j * sig ) / s )

    hh = hh * 0.4/t
    ll = ll * 0.4/t
    kk = kk * 0.4/t

    return hh,ll,kk
 

def complex_rigidity(s,mu,eta,code,par):
    """
    Computes the frequency-dependent rigidity for a viscoelastic layer.
    """

    if code==0:                     # Fluid
        mu_s = 0                    
    elif code==1:                   # Elastic
        mu_s = mu
    elif code==2:                   # Maxwell
        mu_s = mu * s / ( s + mu/eta )
    elif code==3:                   # Newton
        mu_s = eta * s
    elif code==4:                   # Kelvin
        mu_s = mu + eta * s
    elif code==5:                   # Burgers
        mu2  = par[0] * mu
        eta2 = par[1] * eta
        mu_s = mu * s * ( s + mu2/eta2 ) / \
	       ( s**2 + s*(mu/eta + (mu+mu2)/eta2) + (mu*mu2)/(eta*eta2) )   
    elif code==6:                   # Andrade
        alpha = par[0]
        zeta  = par[1]
        gam   = par[2]
        mu_s  = 1/mu + 1/(eta*s) + gam * (1/mu) * (s*eta/mu*zeta)**(-alpha)
        mu_s  = 1/mu_s
    elif code==7:                   # Sundberg-Cooper
        mu2   = par[0] * mu
        eta2  = par[1] * eta
        alpha = par[2]
        zeta  = par[3]
        gam   = par[4]
        mu_s  = mu * s * (s + mu2/eta2) / \
           ( s**2 * (1 + mu**alpha * gam * (s*eta*zeta)**(-alpha)) + \
             s * (mu/eta + (mu+mu2)/eta2 + mu2/eta2 * mu**alpha * gam * (s*eta*zeta)**(-alpha) ) + \
             mu*mu2 / (eta*eta2) )
    elif code==8:                   # Extended Burgers
        # We use eqs (19-21) in Ivins et al 2022
        alpha = par[0]
        delta = par[1]
        tauL  = par[2]
        tauH  = par[3]
        tauM  = eta/mu
        st    = tauM*s

        hypL  = hyp2f1(1,1+alpha,2+alpha, -s*tauL)                
        hypH  = hyp2f1(1,1+alpha,2+alpha, -s*tauH)

        B     = alpha/(1+alpha) * delta * \
                ( (tauL/tauM) / ( (tauH/tauL)**alpha - 1 ) * hypL - (tauH/tauM) / ( 1 - (tauL/tauH)**alpha ) * hypH )               
        D     = 1/( 1 + (1 + delta) * st + st**2 * B )

        mu_s  = mu * st * D

    else:
        raise ValueError("Invalid rheology code: "+str(code))

    return mu_s

def build_model(r, rho, mu, lam, eta, rheology, params, timeunits='kyr'):
    """
    Validates the model, normalizes the parameters and computes 
    some additional physical quantities.
    """

    # ----- Define the gravitational constant

    Gnwt  = 6.674e-11

    # ----- if we have a one-layer model, convert parameters to arrays

    if type(r) in [float, int, np.float64]:
        r = [ r ]
    if type(rho) in [float, int, np.float64]:
        rho = [ rho ]
    if type(mu) in [float, int, np.float64]:
        mu = [ mu ]
    if type(lam) in [float, int, np.float64]:
        lam = [ lam ]
    if type(eta) in [float, int, np.float64]:
        eta = [ eta ]
    if type(rheology) in [str, int]:
        rheology = [ rheology ]

    # ----- If the model is incompressible, set lambda to zeros

    if type(lam)==type(None):
        lam = np.zeros( len(r) )
        comp = False
    else:
        comp = True

    # ----- Check argument types

    for arg in [r, rho, mu, lam, eta, rheology, params]:
        if type(arg) not in [list, tuple, np.ndarray]:
            raise TypeError("Invalid argument type")

    # ----- Convert arguments to float arrays
        
    r   = np.array( r,   dtype=np.float64 )
    rho = np.array( rho, dtype=np.float64 )
    mu  = np.array( mu,  dtype=np.float64 )
    lam = np.array( lam, dtype=np.float64 )
    eta = np.array( eta, dtype=np.float64 )

    params = np.array( params, dtype=np.float64 )

    # ----- Check consistency of input argument size

    for arg in [r,rho,mu,lam,eta]:
        if arg.ndim != 1:
            raise TypeError("Error: r, rho, mu, lam, eta must be 1D vectors")
    
    nla = r.size
    for arg in [r, rho, mu, lam, eta, rheology]:
        if( len(arg) != nla ):
            raise ValueError("Argument size mismatch")

    # If we have only one layer, we check if we need to reshape the parameters vector
        
    if (nla==1) and (params.ndim==1):
        params = np.array( [params] )

    junk=params.shape
    if junk[0]!=nla:
        raise TypeError("Error: the size of the params array is not consistent")

    # ----- Parse the rheology

    # Parse rheology

    rheol = []
    for rcode in rheology:
        if type(rcode) in [ str, np.str_ ]:
            if rcode.lower()=='fluid':
                rheol.append(0)
            elif rcode.lower()=='elastic':
                rheol.append(1)
            elif rcode.lower()=='maxwell':
                rheol.append(2)
            elif rcode.lower()=='newton':
                rheol.append(3)
            elif rcode.lower()=='kelvin':
                rheol.append(4)
            elif rcode.lower()=='burgers':
                rheol.append(5)
            elif rcode.lower()=='andrade':
                rheol.append(6)
            elif (rcode.lower()=='sundberg') or (rcode.lower()=='sundcoop'):
                rheol.append(7)
            elif (rcode.lower()=='ebm'):
                rheol.append(8)
            else:
                raise ValueError('Unknown rheology name "'+rcode+'"')
        elif type(rcode)==int:
            if( (rcode>=0) & (rcode<=8) ):
                rheol.append(rcode)
            else:
                raise ValueError('Invalid rheology code '+str(rcode))
        else:
            raise TypeError('Invalid rheology argument')

    # ----- Pre-compute gamma(alpha+1) for the Andrade and Sundberg-Cooper layers

    for i in range(nla):
        if rheol[i]==6:
            if params[i,1]==0:   # Assume zeta=1 as default
                params[i,1]=1.0
            params[i,2] = gamma(params[i,0]+1)
        if rheol[i]==7:
            params[i,4] = gamma(params[i,0]+1)
        if rheol[i]==8:
            if timeunits=='yr':
                params[i,2] = params[i,2]/1000.
                params[i,3] = params[i,3]/1000.
            elif timeunits=='day':
                params[i,2] = params[i,2]/(365.25*1000.)
                params[i,3] = params[i,3]/(365.25*1000.)
            elif timeunits=='hr':
                params[i,2] = params[i,2]/(24*365.25*1000.)
                params[i,3] = params[i,3]/(24*365.25*1000.)

                  
    # ----- Define reference scales
 
    r0    = r[-1]
    rho0  = rho.max()
    mu0   = mu.max()
    t0    = 1000 * 365.25 * 24 * 3600
    eta0  = mu0 * t0
    mass0 = rho0 * r0**3

    # ----- Normalize model parameters

    r      = r   / r0
    rho    = rho / rho0
    mu     = mu  / mu0
    lam    = lam / mu0
    eta    = eta / eta0

    # ----- Normalized Newton constant

    G      = Gnwt * rho0**2 * r0**2 / mu0

    # ----- Compute the mass of the layers

    mlayer = np.zeros( nla, dtype=np.float64 )

    mlayer[0] = rho[0] * r[0]**3
    for i in range(1,nla):
        mlayer[i] = rho[i] * ( r[i]**3 - r[i-1]**3 )
    mlayer = mlayer * 4 / 3 * pi

    mass = mlayer.sum()

    # ----- Compute gravity at the interface boundaries

    grav = np.zeros( nla, dtype=np.float64 )

    for i in range(nla):
        grav[i] = G * sum( mlayer[0:i+1] ) / r[i]**2

    Model = namedtuple("Model", "r rho mu lam eta rheology params G gravity mass compressible")
    model = Model(r, rho, mu, lam, eta, rheol, params, G, grav, mass, comp)

    return model

def gr_average(r,rho,G):
    """
    Computes the average g/r ratio in each layer.
    """

    # Computes the average g/r ratio in each layer

    nla = len(r)
    xiavg  = np.zeros( nla, dtype=np.float64 );

    # Mass of the layers

    mlayer = np.zeros( nla, dtype=np.float64 )

    mlayer[0] = rho[0] * r[0]**3
    for i in range(1,nla):
        mlayer[i] = rho[i] * ( r[i]**3 - r[i-1]**3 )
    mlayer = mlayer * 4 / 3 * pi

    # average g/r

    xiavg[0] = 4/3 * pi * G * rho[0]
    for i in range(1,nla):
        dM = np.sum(mlayer[0:i]) - 4/3*pi*rho[i]*r[i-1]**3
        # standard average
        xiavg[i] = G*dM/2 * (r[i]+r[i-1]) / ( r[i]**2 * r[i-1]**2 ) + 4/3 * pi * rho[i] * G
        # volume average
        #xiavg[i] = 3*G*dM * np.log(r[i]/r[i-1]) / ( r[i]**3 * r[i-1]**3 ) + 4/3 * pi * rho[i] * G

    return xiavg

def love_numbers_spectrum(n,s,iload,model,xi=1e-4,n0=20,numint=None,det=False,mu_fluid=1e-5):
    """
    Computes the Love numbers spectrum at a given value of the Laplace variable s.
    """

    # Samples the LNs at a given value of the Laplace variable 's'

    # ----- Unpack the model parameters

    r, rho, mu, lam, eta, rheol, rpar, G, grav, mass, comp = model
    nla   = len(r)

    # ----- If the numint flag is not set, use the default
    
    if (numint!=True) and (numint!=False):
        if comp:
            numint=True
        else:
            numint=False

    # ----- Pre-compute the complex rigidity at each layer

    mu_s = np.zeros( nla, dtype=np.complex128 )
    for i in range(nla):
        vec = rpar[i,:]
        mu_s[i] = complex_rigidity(s,mu[i],eta[i],rheol[i],vec)

    # For an incompressible model, lam (and hence lam_s) is a 
    # dummy vector set to zero
    if comp:
        lam_s = np.zeros( nla, dtype=np.complex128 )
        kappa = lam + 2/3*mu
        for j in range(nla):
            lam_s[j] = kappa[j] - 2/3*mu_s[j]
    else:
        lam_s = np.zeros(nla)

    # ----- If needed, we start the propagation of the solution at a 
    #       radius such that stability of propagators is improved

    if n>n0:
        # Find the initial radius
        r0 = r[-1] * ( xi**(1/n) )
        if r0 > r[0]:      
            # Find the layer containing the initial radius   
            k0 = np.where( r > r0 )[0][0]
            # Compute volume-averages for r<=r0
            rhoavg, muavg, lamavg = volume_average(r,rho,mu_s,lam_s,r0)
            # Set the homogeneous sphere parameter to those of the layer
            # where we start integration
            #rhoavg, muavg, lamavg = rho[k0], mu_s[k0], lam_s[k0]
            # Redefine the model by addig a uniform sphere for r<=r0
            r    = np.insert(r[k0:],     0, r0)
            rho  = np.insert(rho[k0:],   0, rhoavg)
            #mu_s = np.insert(mu_s[k0:],  0, muavg)
            #lam_s= np.insert(lam_s[k0:], 0, lamavg)
            nla  = len(r)
            # The uniform sphere is assumed with the same rheology of its outer layer
            rheol = [ rheol[k0] ] + rheol[k0:]
            mu_s = np.insert(mu_s[k0:],  0, mu_s[k0])
            lam_s= np.insert(lam_s[k0:], 0, lam_s[k0])
            # We adjust the gravity acceleration
            g0   = 4/3 * pi * G * rhoavg * r0
            grav = np.insert(grav[k0:], 0, g0)
 
    # ----- Use a fluid rheology on layers close to their fluid limit

    if numint:
        for i in range(nla):
            if abs(mu_s[i])<mu_fluid:
                rheol[i]=0

    # ----- Compute the mass of the layers

    mlayer = np.zeros( nla )
    mlayer[0] = 4/3*pi*r[0]**3 * rho[0]
    for i in range(1,nla):
        mlayer[i] = 4/3*pi*rho[i]*(r[i]**3 - r[i-1]**3)

    # ----- For compressible models, compute the average g/r in each layer
            
    if comp:
        xiavg = gr_average(r,rho,G)

    # ----- Initialize vectors used to propagate the solution

    yf = np.zeros( 2, np.complex128 )
    ys = np.zeros( (6, 3), dtype=np.complex128 )

    # ----- Set the initial conditions at the most internal interface
    
    if rheol[0]==0:          # The inner layer is fluid
        # We rescale the solution by a factor r[0]**n
        yf[0] = 1.0
        yf[1] = 2*(n-1)*r[0]**(-1)
        #yf[0] = r[0]**n
        #yf[1] = 2*(n-1)*r[0]**(n-1)
    else:                    # The inner layer is solid
        if comp:
            #Ydir = direct_compressible_matrix(n,r[0],rho[0],mu_s[0],lam_s[0],xiavg[0],G)
            Ydir = compressible_homogeneous_sphere(n,r[0],rho[0],mu_s[0],lam_s[0],G,rescale=True)
        else:
            Ydir = direct_matrix(n,r[0],rho[0],mu_s[0],grav[0],G)
        for i in range(3):
            ys[:,i] = Ydir[:,i]
        # Rearrange the order of eigensolutions because we do not want
        # to have the 4th component of y3 equal to zero in case the
        # next layer is fluid
        ys = ys[:,[2, 0, 1]]

    # ----- Propagate the eigenfunctions to the surface

    for j in range(1,nla):
    
        if( (rheol[j]!=0) and (rheol[j-1]==0) ):
            
            # Transition from fluid to solid

            ys[:,:] = 0.

            ys[0,0] = -yf[0] / grav[j-1]
            ys[4,0] =  yf[0]
            ys[5,0] =  yf[1]

            ys[1,1] =  1.0
    
            ys[0,2] =  1.0
            ys[2,2] =  rho[j-1] * grav[j-1]
            ys[5,2] =  4 * pi * G * rho[j-1]
    
        elif( (rheol[j]==0) and (rheol[j-1]!=0) ):
                
            # Transition from solid to fluid
            
            z = np.zeros( (6, 2), dtype=np.complex128 )

            z[:,0] = ys[:,0] - ys[3,0]/ys[3,2] * ys[:,2]
            z[:,1] = ys[:,1] - ys[3,1]/ys[3,2] * ys[:,2]
    
            w = z[:,0] - ( z[0,0] + z[4,0]/grav[j-1] - z[2,0]/(rho[j]*grav[j-1]) ) / \
                         ( z[0,1] + z[4,1]/grav[j-1] - z[2,1]/(rho[j]*grav[j-1]) ) * z[:,1]
    
            yf[0] = w[4]
            yf[1] = w[5] - 4 * pi * G * w[2]/grav[j-1]
    
        if( rheol[j]==0 ):

            # Propagate the solution in a fluid layer
    
            Wdir = direct_fluid_matrix( n,r[j  ],rho[j],grav[j  ],G)
            Winv = inverse_fluid_matrix(n,r[j-1],rho[j],grav[j-1],G)
    
            phi = np.matmul(Wdir,Winv)
            yf  = np.matmul(phi,yf)
    
        else:                          
            
            # Propagate the solution in a solid layer
            
            if numint:
                f_g = lambda x: G * ( np.sum(mlayer[0:j]) + 4*pi/3*(x**3 - r[j-1]**3) * rho[j])/x**2
                if comp:
                    f_y   = lambda x, y: np.matmul(Acomp(x,n,rho[j],mu_s[j],lam_s[j],f_g(x),G),y)
                    f_jac = lambda x, y: Acomp(x,n,rho[j],mu_s[j],lam_s[j],f_g(x),G)
                else:
                    f_y   = lambda x, y: np.matmul(Ainc(x,n,rho[j],mu_s[j],f_g(x),G),y)
                    f_jac = lambda x, y: Ainc(x,n,rho[j],mu_s[j],f_g(x),G)
                for i in range(3):
                    yinit = ys[:,i]
                    sol=solve_ivp(f_y,[r[j-1], r[j]],yinit, method='DOP853',rtol=1e-10,atol=1e-10)
                    #sol=solve_ivp(f_y,[r[j-1], r[j]],yinit, method='DOP853',rtol=1e-6,atol=1e-8)
                    #sol=solve_ivp(f_y,[r[j-1], r[j]],yinit, method='RK45',rtol=1e-6,atol=1e-8)
                    #sol=solve_ivp(f_y,[r[j-1], r[j]],yinit, jac=f_jac, method='BDF',rtol=1e-6,atol=1e-8)
                    ys[:,i] = sol.y[:,-1]
            else:
                if comp:
                    Ydir = direct_compressible_matrix( n,r[j  ],rho[j],mu_s[j],lam_s[j],xiavg[j],G)
                    Yinv = direct_compressible_matrix( n,r[j-1],rho[j],mu_s[j],lam_s[j],xiavg[j],G)
                    Yinv = np.linalg.inv(Yinv)
                else:
                    Ydir = direct_matrix( n,r[j  ],rho[j],mu_s[j],grav[j  ],G)
                    Yinv = inverse_matrix(n,r[j-1],rho[j],mu_s[j],grav[j-1],G)
                
                phi = np.matmul(Ydir,Yinv)
                ys  = np.matmul(phi,ys)

            junk=0.   # dummy statement, just to set a breakpoint
    
            
    # ---- Apply the boundary conditions at the surface
    #      and solve the system
    
    x = np.zeros( 3, dtype=np.complex128 )

    if( rheol[-1]==0 ):

        # surface BCs for a fluid
    
        qtidal = -4 * pi * G * ( 2*n+1 ) / (4*pi*r[-1]**2)
        c = qtidal/yf[1]
    
        x[0]=-c*yf[0]/grav[-1]
        x[1]= 0.0
        x[2]= c*yf[0]
    
    else:                            
        
        # surface BCs for a solid
    
        bs = surface_bc(n,r[-1],grav[-1],iload,G)
        
        rr = np.zeros( (3, 3), dtype=np.complex128 )

        if (n==1):
           rr[0,:] = ys[2,:]
           rr[1,:] = ys[3,:]
           rr[2,:] = ys[4,:]
        else:
           rr[0,:] = ys[2,:]
           rr[1,:] = ys[3,:]
           rr[2,:] = ys[5,:]

        c = np.linalg.solve(rr,bs)
        y = np.matmul(ys,c)

        x[0] = y[0]
        x[1] = y[1]
        x[2] = y[4]
    
    hh = x[0] * mass / r[-1]
    ll = x[1] * mass / r[-1]
    kk = ( -1 -x[2] * mass / ( r[-1] * grav[-1] ) )
    
    if det:
        if rheol[-1]==0:
            raise ValueError('Error: cannot compute the secular determinant if the outer layer is fluid')
        return np.linalg.det(rr)
    else:
        return hh, ll, kk
        
def love_numbers(r,rho,mu,lam,eta,rheology,params,degrees,timesteps,loadtype,analysis, \
                 verbose=False, order=8, xi=1e-4, n0=20, numint=None,adaptive=True, \
                 r_sample=[],inversion='default',timeunits='kyr'):
    """
    Computes the Love Numbers.
    """

    # Validate and normalize the model parameters

    model = build_model(r, rho, mu, lam, eta, rheology, params, timeunits=timeunits)
    nla = len(model.r)

    # If the numerical integration / analytical propagation has not been selected,
    # we assume as a default numerical integration for compressible models and
    # analytical propagation for incompressible models

    if (numint!=True) and (numint!=False):
        if model.compressible:
            numint=True
        else:
            numint=False

    # We always disable adaptive PW inversion if the max_order is <=3

    if (adaptive!=True) and (adaptive!=False):
        raise TypeError('Invalid option for the adaptive inversion')
    else:
        if (order<=3):
            adaptive=False

    # Set the load type ('tidal' or 'loading')

    if loadtype.lower()=='tidal':
        iload=0
    elif loadtype.lower()=='loading':
        iload=1
    else:
        raise ValueError( 'Unknown load type "'+loadtype+'"' )

    # Set the analysis type

    if analysis.lower()=='elastic':
        itype=1
    elif analysis.lower()=='fluid':
        itype=2
    elif analysis.lower()=='fluidlimit':
        itype=3
    elif analysis.lower()=='frequency':
        itype=4
    elif analysis.lower()=='laplace':
        itype=5    
    elif analysis.lower() in ['timedomain', 'heaviside']:
        itype=6
    elif analysis.lower()=='collocation':
        itype=7
    elif analysis.lower()=='secular':
        itype=8
    else:
        raise ValueError('Unknown analysis type "'+analysis+'"' )
 
    # We cannot compute loading LNs if the outer layer is fluid

    if( (iload==1) and ((itype==2) or (model.rheology[-1]==0)) ):
        raise TypeError("Loading BCs for a fluid layer are not supported")

    # For a time domain analysis we check the inversion method

    if itype==6:
        if type(inversion)!=str:
            raise TypeError("Invalid type for the inversion method")
        if inversion.lower()=='postwidder':
            iinv=1
        elif inversion.lower()=='talbot':
            iinv=2
        elif inversion.lower()=='default':
            # As a default we use the Fixed-Talbot inversion
            iinv=2
            # # As a default we use Post-Widder for incompressible models
            # # and Talbot for compressible models
            # if model.compressible:
            #     iinv=2
            # else:
            #     iinv=1
        else:
            raise ValueError('Unknown inversion method "'+inversion+'"')

    # Number of time steps and of harmonic degrees
    # If we have only one timestep or harmonic degree, we reshape
    # it as a vector

    if type(timesteps) in [int, float]:
        timesteps = [ timesteps ]

    if type(degrees)==int:
        degrees = [ degrees ]

    nt   = len(timesteps)
    ndeg = len(degrees)

    # Convert the timesteps to a float numpy array

    t = np.array( timesteps, dtype=np.float64 )

    # Convert times to kyr, if needed

    if timeunits!='kyr':
        if timeunits=='yr':
            t = t / 1000.0
        elif timeunits=='day':
            t = t / (365.25 * 1000.0)
        elif timeunits=='hr':
            t = t / (24 * 365.25 * 1000.0)
        else:
            raise ValueError('Unknown time units "'+timeunits+'". Valid ones are kyr,yr,day,hr')

    # For elastic and fluid analyses we will ingnore the timesteps

    if (itype==1) or (itype==2) or (itype==3):
        nt = 1

    # Adjust the rheology vector for some specific analyses

    if itype==1:
        # Elastic analysis
        for i in range(nla):
            if model.rheology[i]!=0:
                model.rheology[i]=1
    elif itype==2:
        # Fluid analysis
        for i in range(nla):
            model.rheology[i]=0
    elif itype==3:
        # viscoelastic fluid limit
        for i in range(nla):
            if model.rheology[i]!=1:
                model.rheology[i]=0
        
    # For a time-domain analysis, compute the Salzer weights

    if (itype==6):
        if (iinv==1):
            if adaptive:
                zeta = []
                for m in range(1,order+1):
                    zeta.append( salzer_weights(m) )
            else:
                zeta = salzer_weights(order)

    # For a collocation analysis we need some additional setup
    
    if itype==7:

        # Vectors for the elastic LNs

        he = np.zeros( ndeg, dtype=np.float64 )
        le = np.zeros( ndeg, dtype=np.float64 )
        ke = np.zeros( ndeg, dtype=np.float64 )
        
        # Vectors for the Laplace samples

        hv = np.zeros( nt, dtype=np.float64 )
        lv = np.zeros( nt, dtype=np.float64 )
        kv = np.zeros( nt, dtype=np.float64 )

        # Build the 'chi' matrix

        chi = np.zeros( (nt,nt), dtype=np.float64 )
        
        for i in range(nt):
            for j in range(nt):
                chi[i,j] = 1.0 / ( 1.0/t[i] + 1.0/t[j] )

        # Save the original rheology

        r_save = [0] * nla
        r_save[:] = model.rheology[:]

        # Build an elastic version of the rheology

        r_ela = r_save.copy()
        for i in range(nla):
            if r_ela[i]!=0:
                r_ela[i]=1
        
    # ------- Allocate output arrays

    det_s  = np.zeros( (ndeg, nt), dtype=np.complex128 )
    h_love = np.zeros( (ndeg, nt), dtype=np.complex128 )
    l_love = np.zeros( (ndeg, nt), dtype=np.complex128 )
    k_love = np.zeros( (ndeg, nt), dtype=np.complex128 )
    
    # ------ Initial time-mark

    t1 = time.perf_counter()
    
    # ------- Compute LNs

    for idx_n in range(ndeg):
       
        n = degrees[idx_n]

        if (itype==1) or (itype==2) or (itype==3):

            # Elastic / full fluid limit / viscoelastic fluid limit

            s = 1.0   # a dummy value

            hh, ll, kk = love_numbers_spectrum(n,s,iload,model, xi=xi, n0=n0, numint=numint)

            h_love[idx_n,0] = hh
            l_love[idx_n,0] = ll
            k_love[idx_n,0] = kk

        elif ( itype==4 ):
            
            # Frequency-domain analysis
            
            for i in range(nt):

                omega = 2 * pi / t[i]
                s     = 1j * omega
            
                hh, ll, kk = love_numbers_spectrum(n,s,iload,model, xi=xi, n0=n0, numint=numint)
            
                h_love[idx_n,i] = hh
                l_love[idx_n,i] = ll
                k_love[idx_n,i] = kk
            
        elif ( itype==5 ):
            
            # Laplace spectral analysis
            
            for i in range(nt):
            
                s = 1.0/t[i]
                
                hh, ll, kk = love_numbers_spectrum(n,s,iload,model, xi=xi, n0=n0, numint=numint)

                h_love[idx_n,i] = hh
                l_love[idx_n,i] = ll
                k_love[idx_n,i] = kk
        
        elif ( itype==6 ):
            
            # Time-domain analysis
            
            #for i in range(nt):
            #          
            #    f = math.log(2.0) / t[i]
            #
            #    for k in range(2*order):
            #
            #        s = f * (k+1)      # k goes from 0 to 2*order-1
            #        
            #        hh, ll, kk = love_numbers_spectrum(n,s,iload,model, xi=xi, n0=n0, numint=numint)
            #        
            #        h_love[idx_n,i] += hh * zeta[k] * f / s
            #        l_love[idx_n,i] += ll * zeta[k] * f / s
            #        k_love[idx_n,i] += kk * zeta[k] * f / s

            if iinv==1:

                if adaptive:
                    for i in range(nt):
                        h_love[idx_n,i], l_love[idx_n,i], k_love[idx_n,i] = \
                            adaptive_postwidder_inversion(n,t[i],iload,model,order,zeta,xi,n0,numint)
                else:    
                    for i in range(nt):
                        h_love[idx_n,i], l_love[idx_n,i], k_love[idx_n,i] = \
                            postwidder_inversion(n,t[i],iload,model,order,zeta,xi,n0,numint)
                        
            elif iinv==2:
        
                for i in range(nt):
                    h_love[idx_n,i], l_love[idx_n,i], k_love[idx_n,i] = \
                        talbot_inversion(n,t[i],iload,model,order,xi,n0,numint)

        elif ( itype==7 ):
            
            # Collocation analysis
            
            # ---- Compute the elastic LNs
                   
            s = 1.0

            model.rheology[:] = r_ela[:]

            hh, ll, kk = love_numbers_spectrum(n,s,iload,model, xi=xi, n0=n0, numint=numint)

            he[idx_n] = np.real(hh)
            le[idx_n] = np.real(ll)
            ke[idx_n] = np.real(kk)

            # ---- Sample the viscoelastic LNs
            
            model.rheology[:] = r_save[:]
            
            for i in range(nt):

                s = 1.0 / t[i]

                hh, ll, kk = love_numbers_spectrum(n,s,iload,model, xi=xi, n0=n0, numint=numint)

                hv[i] = np.real(hh) - he[idx_n]
                lv[i] = np.real(ll) - le[idx_n]
                kv[i] = np.real(kk) - ke[idx_n]

            h_love[idx_n,:] = np.linalg.solve(chi,hv)
            l_love[idx_n,:] = np.linalg.solve(chi,lv)
            k_love[idx_n,:] = np.linalg.solve(chi,kv)

        elif (itype==8):

            # Secular determinant analysis

            for i in range(nt):
            
                s = 1.0/t[i]
                
                det_s[idx_n,i] = love_numbers_spectrum(n,s,iload,model, xi=xi, n0=n0, numint=numint, det=True)
        
        if verbose:
            t2 = time.perf_counter()
            #print( " - Harmonic degree n = " + str(n) + " ( " + str(t2-t1) + " s )" )
            print( " - Harmonic degree n = {:d} ({:.4f}s)".format(n,t2-t1) )
            t1 = t2

    # We drop the imaginary part except for frequency-domain analysis            
    if itype != 4:
        h_love = np.real(h_love)
        l_love = np.real(l_love)
        k_love = np.real(k_love)

    # We drop the time dimensions for elastic and fluid analyses
    if (itype==1) or (itype==2) or (itype==3):
        h_love = np.squeeze(h_love)
        l_love = np.squeeze(l_love)
        k_love = np.squeeze(k_love)

    if itype == 7:
        return he, le, ke, h_love, l_love, k_love
    elif itype == 8:
        return det_s
    else:
        return h_love, l_love, k_love
 
def volume_average(r,rho,mu,lam,r0):
    """
    Computes the volume averages of rho,mu,eta for r<=r0.
    """

    # Top and bottom radii
    rtop = r
    rbot = np.insert( r[:-1], 0, 0.0 )

    # Find the layer containing r0
    k0 = np.where( r > r0 )[0][0]
    
    rhoavg = 0.
    muavg  = 0.
    lamavg = 0.
    vol    = 0.

    for k in range(k0):
        vl = 4/3 * pi * (rtop[k]**3 - rbot[k]**3)
        muavg  += mu[k]  *  vl
        lamavg += lam[k] *  vl
        rhoavg += rho[k] *  vl
        vol    += vl
    
    vl = 4/3 * pi * (r0**3 - rbot[k0]**3)
    muavg  += mu[k0]  * vl
    lamavg += lam[k0] * vl
    rhoavg += rho[k0] * vl
    vol    += vl

    rhoavg = rhoavg / vol
    muavg  = muavg  / vol
    lamavg = lamavg / vol


    return rhoavg, muavg, lamavg

def sph_bessel_j(n,z):
    """
    Computes the jn spherical Bessel function and its first and
    second derivatives.
    """

    # j'(n,z) = j(n-1,z) - (n+1)/z j(n,z)

    # j''(n,z) = j'(n-1,z) + (n+1)/z**2 j(n,z) - (n+1)/z j'(n,z) =
    #            j'(n-1,z) + (n+1)/z * ( j(n,z)/z - j'(n,z) )

    jn   = lambda n,z : spherical_jn(n,z)
    djn  = lambda n,z : jn(n-1,z) - (n+1)/z * jn(n,z)
    d2jn = lambda n,z : djn(n-1,z) + (n+1)/z * ( jn(n,z)/z - djn(n,z) )

    if n==1:
        # For n=1 we use the explicit formulas
        sz, cz = np.sin(z), np.cos(z)
        j0 = sz/z**2 - cz/z
        j1 = 2*cz/z**2 - 2*sz/z**3 + sz/z
        j2 = -6*cz/z**3 + cz/z + 6*sz/z**4 - 3*sz/z**2
    else:
        j0 = jn(n,z)
        j1 = djn(n,z)
        j2 = d2jn(n,z)

    return j0,j1,j2

def sph_bessel_y(n,z):
    """
    Computes the yn spherical Bessel function and its first and
    second derivatives.
    """

    # y'(n,z) = y(n-1,z) - (n+1)/z y(n,z)

    # y''(n,z) = y'(n-1,z) + (n+1)/z**2 y(n,z) - (n+1)/z y'(n,z) =
    #            y'(n-1,z) + (n+1)/z * ( y(n,z)/z - y'(n,z) )

    yn   = lambda n,z : spherical_yn(n,z)
    dyn  = lambda n,z : yn(n-1,z) - (n+1)/z * yn(n,z)
    d2yn = lambda n,z : dyn(n-1,z) + (n+1)/z * ( yn(n,z)/z - dyn(n,z) )

    if n==1:
        # For n=1 we use the explicit formulas
        sz, cz = np.sin(z), np.cos(z)  
        y0 = -cz/z**2 - sz/z
        y1 = 2*cz/z**3 - cz/z + 2*sz/z**2
        y2 = -6*cz/z**4 + 3*cz/z**2 - 6*sz/z**3 + sz/z
    else:      
        y0 = yn(n,z)
        y1 = dyn(n,z)
        y2 = d2yn(n,z)

    return y0,y1,y2

def direct_compressible_matrix(n,r,rho,mu,lam,xi,G):
    """
    Computes the compressible fundamental matrix in a given layer.
    """

    #xi = 4/3 * pi * G * rho
    #xi = gra/r
    vp = sqrt( (lam + 2*mu)/rho )
    vs = sqrt( mu/rho )

    #alpha = 2*sqrt(xi)/vp  # valid only for a uniform sphere
    alpha = sqrt(4*pi*G*rho + xi)/vp
    gamma = 2*xi*sqrt(n*(n+1))/(vp*vs)
    ksq =  ( alpha**2 + sqrt( alpha**4 + gamma**2 ) ) / 2 
    qsq =  ( alpha**2 - sqrt( alpha**4 + gamma**2 ) ) / 2 

    kr  = sqrt(ksq)*r
    k2r = ksq*r
    c   = -xi / (ksq * vs**2)

    jn,j1n,j2n = sph_bessel_j(n,kr)

    y1 = np.array( [ \
        -( n*(n+1) * c * jn + kr * j1n ) / k2r, \
        -( (1+c) * jn + c * kr * j1n) / k2r, \
        lam*jn + 2*mu*( n*(n+1) * c / kr * ( jn/kr - j1n ) - j2n ), \
        -mu * c * jn + 2*mu*( (1+c)/kr * (jn/kr - j1n) - c * j2n ), \
        #3*xi * jn / ksq, \
        #3*xi * (1 - n*c) * (1 + n) / k2r * jn \
        4*pi*G*rho * jn / ksq, \
        4*pi*G*rho * (1 - n*c) * (1 + n) / k2r * jn \
        ], dtype=np.complex128)
    
    qr  = sqrt(qsq)*r
    q2r = qsq*r
    c   = -xi / (qsq * vs**2)

    jn,j1n,j2n = sph_bessel_j(n,qr)

    y2 = np.array( [ \
        -( n*(n+1) * c * jn + qr * j1n ) / q2r, \
        -( (1+c) * jn + c * qr * j1n) / q2r, \
        lam*jn + 2*mu*( n*(n+1) * c / qr * ( jn/qr - j1n ) - j2n ), \
        -mu * c * jn + 2*mu*( (1+c)/qr * (jn/qr - j1n) - c * j2n ), \
        #3*xi * jn / qsq, \
        #3*xi * (1 - n*c) * (1 + n) / q2r * jn \
        4*pi*G*rho * jn / qsq, \
        4*pi*G*rho * (1 - n*c) * (1 + n) / q2r * jn \
        ], dtype=np.complex128)
    
    # CHANGED on 19-NOV-2024
    # The following is the expression in Vermeersen 1996
    #y3 = np.array( [ \
    #    n * r**(n-1),
    #    r**(n-1),
    #    2*mu*n*(n-1) * r**(n-2),
    #    2*mu*(n-1) * r**(n-2),
    #    #-xi   * n*r**n,
    #    #-2*xi * n*(n-1) * r**(n-1) \
    #    -4/3*pi*G*rho * n*r**n,
    #    -8/3*pi*G*rho * n*(n-1) * r**(n-1) \
    #    ], dtype=np.complex128)
    
    # This is instead the expression that I found analytically, and that
    # coincides with the one above only if xi = 4/3pi G rho, i.e. for a 
    # homogeneous sphere
    y3 = np.array( [ \
        n * r**(n-1),
        r**(n-1),
        2*mu*n*(n-1) * r**(n-2),
        2*mu*(n-1) * r**(n-2),
        -xi * n*r**n,
        -n*((2*n+1)*xi - 4*pi*G*rho) * r**(n-1) \
        ], dtype=np.complex128)

    c   = -xi / (ksq * vs**2)

    yn,y1n,y2n = sph_bessel_y(n,kr)

    y4 = np.array( [ \
        -( n*(n+1) * c * yn + kr * y1n ) / k2r, \
        -( (1+c) * yn + c * kr * y1n) / k2r, \
        lam*yn + 2*mu*( n*(n+1) * c / kr * ( yn/kr - y1n ) - y2n ), \
        -mu * c * yn + 2*mu*( (1+c)/kr * (yn/kr - y1n) - c * y2n ), \
        #3*xi * yn / ksq, \
        #3*xi * (1 - n*c) * (1 + n) / k2r * yn \
        4*pi*G*rho * yn / ksq, \
        4*pi*G*rho * (1 - n*c) * (1 + n) / k2r * yn \
        ], dtype=np.complex128)

    c   = -xi / (qsq * vs**2)

    yn,y1n,y2n = sph_bessel_y(n,qr)

    y5 = np.array( [ \
        -( n*(n+1) * c * yn + qr * y1n ) / q2r, \
        -( (1+c) * yn + c * qr * y1n) / q2r, \
        lam*yn + 2*mu*( n*(n+1) * c / qr * ( yn/qr - y1n ) - y2n ), \
        -mu * c * yn + 2*mu*( (1+c)/qr * (yn/qr - y1n) - c * y2n ), \
        #3*xi * yn / qsq, \
        #3*xi * (1 - n*c) * (1 + n) / q2r * yn \
        4*pi*G*rho * yn / qsq, \
        4*pi*G*rho * (1 - n*c) * (1 + n) / q2r * yn \
        ], dtype=np.complex128)

    # FIXED ON 19-DEC-2023
    #y6 = np.array( [ \
    #    -(n+1) / r**(n+2),
    #    1/r**(n+2),
    #    2*mu*(n+1)*(n+2) / r**(n+3),
    #    -2*mu*(n+2) / r**(n+3),
    #    -xi*n/r**(n+1),
    #    -3*xi*(n+1)/r**(n+2) \
    #    ], dtype=np.complex128)    

    # CHANGED ON 19-NOV-2024
    # what follows is the y6 expression found by trial-and-error
    # that correctly solve the system y'=Ay for a homogeneous sphere,
    # as verified numerically
    #y6 = np.array( [ \
    #    -(n+1) / r**(n+2),
    #    1/r**(n+2),
    #    2*mu*(n+1)*(n+2) / r**(n+3),
    #    -2*mu*(n+2) / r**(n+3),
    #    #xi * (n+1)/r**(n+1),
    #    #-3*xi * (n+1)/r**(n+2) \
    #    4/3*pi*G*rho * (n+1)/r**(n+1),
    #    -4*pi*G*rho * (n+1)/r**(n+2) \
    #    ], dtype=np.complex128)    

    # the following expression instead is the solution that I obtained
    # analytically for y6, and which coincides with the one above only if 
    # xi = (4/3)*pi*G*rho (i.e., for a homogeneous sphere)
    y6 = np.array( [ \
        -(n+1) / r**(n+2),
        1/r**(n+2),
        2*mu*(n+1)*(n+2) / r**(n+3),
        -2*mu*(n+2) / r**(n+3),
        xi * (n+1)/r**(n+1),
        -4*pi*G*rho * (n+1)/r**(n+2) \
        ], dtype=np.complex128)    


    Y = np.zeros((6,6), dtype=np.complex128)

    Y[:,0] = y1
    Y[:,1] = y2
    Y[:,2] = y3
    Y[:,3] = y4
    Y[:,4] = y5
    Y[:,5] = y6

    return Y

def Acomp(r,n,rho,mu,lam,g,G):
    """
    Computes the 'A' matrix for the system dy/dr = Ay in the compressible case.
    """
    A = np.zeros((6,6), np.complex128)

    beta = lam+2*mu
    k    = lam+(2/3)*mu

    A[0,0] = -2/r * (lam/beta)
    A[1,0] = -1/r
    A[2,0] = 4/r * (3*mu/r*(k/beta) - rho*g)
    A[3,0] = 1/r * (rho*g - 6*mu/r*(k/beta))
    A[4,0] = -4*pi*G*rho
    A[5,0] = -4*pi*G*rho*(n+1)/r

    A[0,1] = n*(n+1)/r * (lam/beta)
    A[1,1] = 1/r
    A[2,1] = -n*(n+1)/r * (6*mu/r*(k/beta) - rho*g)
    A[3,1] = -2*mu/r**2 * (1-n*(n+1)*(1+lam/beta))
    A[5,1] = 4*pi*G*rho*n*(n+1)/r

    A[0,2] = 1/beta
    A[2,2] = -4/r*(mu/beta)
    A[3,2] = -1/r*(lam/beta)

    A[1,3] = 1/mu
    A[2,3] = n*(n+1)/r
    A[3,3] = -3/r

    A[2,4] = -rho*(n+1)/r
    A[3,4] = rho/r
    A[4,4] = -(n+1)/r

    A[2,5] = rho
    A[4,5] = 1
    A[5,5] = (n-1)/r

    return A 

def Ainc(r,n,rho,mu,gra,G):
    """
    Computes the 'A' matrix for the system dy/dr = Ay in the incompressible case.
    """
    A = np.zeros((6,6), np.complex128)

    A[0,0] = -2/r
    A[1,0] = -1/r
    A[2,0] = 4/r * (3*mu/r - rho*gra)
    A[3,0] = 1/r * (rho*gra - 6*mu/r)
    A[4,0] = -4*pi*G*rho
    A[5,0] = -4*pi*G*rho*(n+1)/r

    A[0,1] = n*(n+1)/r
    A[1,1] = 1/r
    A[2,1] = -n*(n+1)/r * (6*mu/r - rho*gra)
    A[3,1] = -2*mu/r**2 * (1-2*n*(n+1))
    A[5,1] = 4*pi*G*rho*n*(n+1)/r

    A[3,2] = -1/r

    A[1,3] = 1/mu
    A[2,3] = n*(n+1)/r
    A[3,3] = -3/r

    A[2,4] = -rho*(n+1)/r
    A[3,4] = rho/r
    A[4,4] = -(n+1)/r

    A[2,5] = rho
    A[4,5] = 1
    A[5,5] = (n-1)/r

    return A 

def compressible_homogeneous_sphere(n,r,rho,mu,lam,G,rescale=False,order=50):
    """
    Computes the regular part of the compressible fundamental matrix.
    If rescale=True, the y1 and y2 solutions are rescaled by jn(z), and the jn'/jn
    and jn''/jn ratios are evaluated with a continued fraction series of 
    the specified order according to Coyt et al., 2008.
    The y3 solution is instead rescaled by r**n. 
    """


    xi = 4/3 * pi * G * rho
    vp = sqrt( (lam + 2*mu)/rho )
    vs = sqrt( mu/rho )

    #alpha = 2*sqrt(xi)/vp  # valid only for a uniform sphere
    alpha = sqrt(4*pi*G*rho + xi)/vp
    gamma = 2*xi*sqrt(n*(n+1))/(vp*vs)
    ksq =  ( alpha**2 + sqrt( alpha**4 + gamma**2 ) ) / 2 
    qsq =  ( alpha**2 - sqrt( alpha**4 + gamma**2 ) ) / 2 

    kr  = sqrt(ksq)*r
    k2r = ksq*r

    qr  = sqrt(qsq)*r
    q2r = qsq*r

    if rescale:

        c   = -xi / (ksq * vs**2)        
        
        # Evaluate the ratios j'/j and j''/j through continuous fractions
        j1r = n/(kr) - sph_bessel_ratio(n,kr,order)
        j2r = ( n*(n-1)/(ksq * r**2) - 1  ) + 2/(kr) * sph_bessel_ratio(n,kr,order)
        
        y1 = np.array( [ \
            -( n*(n+1) * c + kr * j1r ) / k2r, \
            -( (1+c) + c * kr * j1r) / k2r, \
            lam + 2*mu*( n*(n+1) * c / kr * ( 1/kr - j1r ) - j2r ), \
            -mu * c + 2*mu*( (1+c)/kr * (1/kr - j1r) - c * j2r ), \
            4*pi*G*rho / ksq, \
            4*pi*G*rho * (1 - n*c) * (1 + n) / k2r \
            ], dtype=np.complex128)
        
        c   = -xi / (qsq * vs**2)
        # Evaluate the ratios j'/j and j''/j through continuous fractions
        j1r = n/(qr) - sph_bessel_ratio(n,qr,order)
        j2r = ( n*(n-1)/(qsq * r**2) - 1  ) + 2/(qr) * sph_bessel_ratio(n,qr,order)

        y2 = np.array( [ \
            -( n*(n+1) * c + qr * j1r ) / q2r, \
            -( (1+c) + c * qr * j1r) / q2r, \
            lam + 2*mu*( n*(n+1) * c / qr * ( 1/qr - j1r ) - j2r ), \
            -mu * c + 2*mu*( (1+c)/qr * (1/qr - j1r) - c * j2r ), \
            4*pi*G*rho / qsq, \
            4*pi*G*rho * (1 - n*c) * (1 + n) / q2r \
            ], dtype=np.complex128)
        
        y3 = np.array( [ \
            n / r,
            1 / r,
            2*mu*n*(n-1) / r**2,
            2*mu*(n-1) / r**2,
            -4/3*pi*G*rho * n,
            -8/3*pi*G*rho * n*(n-1) / r \
            ], dtype=np.complex128)

    else:

        c   = -xi / (ksq * vs**2)        
        jn,j1n,j2n = sph_bessel_j(n,kr)

        y1 = np.array( [ \
            -( n*(n+1) * c * jn + kr * j1n ) / k2r, \
            -( (1+c) * jn + c * kr * j1n) / k2r, \
            lam*jn + 2*mu*( n*(n+1) * c / kr * ( jn/kr - j1n ) - j2n ), \
            -mu * c * jn + 2*mu*( (1+c)/kr * (jn/kr - j1n) - c * j2n ), \
            4*pi*G*rho * jn / ksq, \
            4*pi*G*rho * (1 - n*c) * (1 + n) / k2r * jn \
            ], dtype=np.complex128)
    
        c   = -xi / (qsq * vs**2)
        jn,j1n,j2n = sph_bessel_j(n,qr)

        y2 = np.array( [ \
            -( n*(n+1) * c * jn + qr * j1n ) / q2r, \
            -( (1+c) * jn + c * qr * j1n) / q2r, \
            lam*jn + 2*mu*( n*(n+1) * c / qr * ( jn/qr - j1n ) - j2n ), \
            -mu * c * jn + 2*mu*( (1+c)/qr * (jn/qr - j1n) - c * j2n ), \
            4*pi*G*rho * jn / qsq, \
            4*pi*G*rho * (1 - n*c) * (1 + n) / q2r * jn \
            ], dtype=np.complex128)
        
        y3 = np.array( [ \
            n * r**(n-1),
            r**(n-1),
            2*mu*n*(n-1) * r**(n-2),
            2*mu*(n-1) * r**(n-2),
            -4/3*pi*G*rho * n*r**n,
            -8/3*pi*G*rho * n*(n-1) * r**(n-1) \
            ], dtype=np.complex128)
        
    
    Y = np.zeros((6,3), dtype=np.complex128)

    Y[:,0] = y1
    Y[:,1] = y2
    Y[:,2] = y3
 
    return Y

def evaluate_cf( terms ):
    """
    Evaluates a continuous fraction in the form
    b0 + a1 / (b1 + (a2 / (b2 + (a3 / (b3 + ... )))))
    The input shall contain ( b0, (a1,b1), (a2,b2), (a3, b3), ... ).
    """
    n = len(terms)
    s = 0.
    for i in range(n-1,0,-1):
        a = terms[i][0]
        b = terms[i][1]
        s = a/(b+s)
    s += terms[0]
    return s

def sph_bessel_ratio(n,z,order):
    """
    Computes the ratio between spherical Bessel functions 
           j_{n+1}(z) / j_n(z)
     using the continued fraction series by Coyt et al. 2008
    up to a given order, which shall be >= 2.
    """
    seq = [0, (z/(2*n+3),1)]
    for m in range(2,order+1):
        seq.append( ((1j * z)**2/(4*(n+m-1/2)*(n+m+1/2)) ,1) )
    return evaluate_cf(seq)

# -------------------------------------------------------------------------- PREM functions

def rho_prem(rad,ocean=False):
    """
    rho_prem(r,ocean) returns the PREM density in kg/m**3 at radius r (in km)
    ocean=False or True selects continental or oceanic PREM, respectively.
    """
    rt  = 6371
    z = rad/rt

    # Outside the Earth
    if( (rad<0) | (rad>6371) ):
        rhol = 0
    # Inner core
    elif( (rad>=0) & (rad<1221.5) ):
        rhol = 13.0885 - 8.8381*(z**2)
    # Outer core
    elif( (rad>=1221.5) & (rad<3480.0) ):
        rhol = 12.5815 -1.2638*(z) -3.6426*(z**2) -5.5281*(z**3)
    # Lower mantle
    elif( (rad>=3480.0) & (rad<5701.0) ):
        rhol =  7.9565 -6.4761*(z) +5.5283*(z**2) -3.0807*(z**3)
    # Transition zone (I)
    elif( (rad>=5701.0) & (rad<5771.0) ):
        rhol =  5.3197 -1.4836*(z)
    # Transition zone (II)
    elif( (rad>=5771.0) & (rad<5971.0) ):
        rhol = 11.2494 -8.0298*(z)
    # Transition zone (III)
    elif( (rad>=5971.0) & (rad<6151.0) ):
        rhol =  7.1089 -3.8045*(z)
    # LVZ & LID
    elif( (rad>=6151.0) & (rad<6346.6) ):
        rhol =  2.6910 +0.6924*(z)
    # Lower Crust
    elif( (rad>=6346.6) & (rad<6356.0) ):
        rhol = 2.900
    # Continental Upper Crust
    elif( (not ocean) & (rad>=6356.0) & (rad<=6371.0) ):
        rhol = 2.600
    # Oceanic Upper Crust
    elif( (ocean) & (rad>=6356.0) & (rad<=6368.0) ):
        rhol = 2.600
    # Ocean layer
    elif( (ocean) & (rad>=6368.0) & (rad<=6371.0) ):
        rhol = 1.020
  
    rho = rhol * 1000 #Density is now in units of kg/m**3
    return rho

def vp_prem(rad,ocean=False):
    """
    vp_prem(r,ocean) returns the PREM P waves velocity in m/s at radius r (in km)
    ocean=False or True selects continental or oceanic PREM, respectively.
    """
    rt  = 6371
    z = rad/rt

    # Outside the Earth
    if( (rad<0) | (rad>6371) ):
        v = 0
    # Inner core
    elif( (rad>=0) & (rad<1221.5) ):
        v = 11.26220 - 6.3640*(z**2)
    # Outer core
    elif( (rad>=1221.5) & (rad<3480.0) ):
        v = 11.0487 -4.0362*(z) +4.8023*(z**2) -13.5732*(z**3)
    # Lower mantle (I)
    elif( (rad>=3480.0) & (rad<3630.0) ):
        v = 15.3891 -5.3181*(z) +5.5242*(z**2) -2.5114*(z**3)
    # Lower mantle (II)
    elif( (rad>=3630.0) & (rad<5600.0) ):
        v = 24.9520 -40.4673*(z) +51.4823*(z**2) -26.6419*(z**3)
    # Lower mantle (III)
    elif( (rad>=5600.0) & (rad<5701.0) ):
        v = 29.2766 -23.6027*(z) +5.5242*(z**2) -2.5514*(z**3)
    # Transition zone (I)
    elif( (rad>=5701.0) & (rad<5771.0) ):
        v = 19.0957 -9.8672*(z)
    # Transition zone (II)
    elif( (rad>=5771.0) & (rad<5971.0) ):
        v = 39.7027 -32.6166*(z)
    # Transition zone (III)
    elif( (rad>=5971.0) & (rad<6151.0) ):
        v = 20.3926  -12.2596*(z)
    # LVZ & LID
    elif( (rad>=6151.0) & (rad<6346.6) ):
        v = 4.1875 +3.9382*(z)
    # Lower Crust
    elif( (rad>=6346.6) & (rad<6356.0) ):
        v = 6.800
    # Continental Upper Crust
    elif( (not ocean) & (rad>=6356.0) & (rad<=6371.0) ):
        v = 5.800
    # Oceanic Upper Crust
    elif( (ocean) & (rad>=6356.0) & (rad<6368.0) ):
        v = 5.800
    # Ocean layer
    elif( (ocean) & (rad>=6368.0) & (rad<=6371.0) ):
        v = 1.450
    
    vp = v * 1000;    #Vp is now in units of m/s

    return vp

def vs_prem(rad,ocean=False):
    """
    vs_prem(r,ocean) returns the PREM S waves velocity in m/s at radius r (in km)
    ocean=False or True selects continental or oceanic PREM, respectively.
    """
    rt  = 6371
    z = rad/rt

    # Outside the Earth
    if( (rad<0) | (rad>6371) ):
        v = 0
    # Inner core
    elif( (rad>=0) & (rad<1221.5) ):
        v = 3.6678 - 4.4475*(z**2)
    # Outer core
    elif( (rad>=1221.5) & (rad<3480.0) ):
        v = 0
    # Lower mantle (I)
    elif( (rad>=3480.0) & (rad<3630.0) ):
        v = 6.9254 +1.4672*(z) -2.0834*(z**2) +0.9783*(z**3)
    # Lower mantle (II)
    elif( (rad>=3630.0) & (rad<5600.0) ):
        v = 11.1671 -13.7818*(z) +17.4575*(z**2) -9.2777*(z**3)
    # Lower mantle (III)
    elif( (rad>=5600.0) & (rad<5701.0) ):
        v = 22.3459 -17.2473*(z) -2.0834*(z**2) +0.9783*(z**3)
    # Transition zone (I)
    elif( (rad>=5701.0) & (rad<5771.0) ):
        v = 9.9839 -4.9324*(z)
    # Transition zone (II)
    elif( (rad>=5771.0) & (rad<5971.0) ):
        v = 22.3512 -18.5856*(z)
    # Transition zone (III)
    elif( (rad>=5971.0) & (rad<6151.0) ):
        v = 8.9496 -4.4597*(z)
    # LVZ & LID
    elif( (rad>=6151.0) & (rad<6346.6) ):
        v = 2.1519 +2.3481*(z)
    # Lower Crust
    elif( (rad>=6346.6) & (rad<6356.0) ):
        v = 3.900
     # Continental Upper Crust
    elif( (not ocean) & (rad>=6356.0) & (rad<=6371.0) ):
        v = 3.200
    # Oceanic Upper Crust
    elif( (ocean) & (rad>=6356.0) & (rad<6368.0) ):
        v = 3.200
    # Ocean layer
    elif( (ocean) & (rad>=6368.0) & (rad<=6371.0) ):
        v = 0.0
    
    vs = v * 1000    #Vs is now in units of m/s

    return vs


def prem_average(r1, r2, ocean=False):
    """
    rho, mu = prem_average(r1, r2, ocean)
    computes density and rigidity as a PREM average between r1 and r2 (in km).
    ocean=False (True) selects continental (oceanic) PREM.
    """

    import scipy.integrate as integrate

    rhor2 = lambda x: rho_prem(x,ocean) *   x**2
    mur2  = lambda x: rho_prem(x,ocean) *   vs_prem(x,ocean)**2 * x**2
    lamr2 = lambda x: rho_prem(x,ocean) * ( vp_prem(x,ocean)**2 - 2 * vs_prem(x,ocean)**2 ) * x**2 

    avero    = integrate.quad(rhor2,r1,r2, limit=1000)
    avemu    = integrate.quad(mur2, r1,r2, limit=1000) 
    avela    = integrate.quad(lamr2,r1,r2, limit=1000)

    avero    = avero[0]   * 3.0 / (r2**3-r1**3)
    avemu    = avemu[0]   * 3.0 / (r2**3-r1**3)
    avela    = avela[0]   * 3.0 / (r2**3-r1**3)

    return avero, avemu, avela
    
def boussinesq(r,rho,mu,lam,G=6.674e-11):
    """
    Compute asymptotic limits according to Boussinesq's relations
    See Farrell 1972, eq (36).
    """

    a = r[-1]

    m = 0.
    for i in range(len(r)):
        if i==0:
            r1 = 0.
        else:
            r1 = r[i-1]
        r2 = r[i]
        m += 4/3 * pi * rho[i] * (r2**3 - r1**3)
    
    g = G*m/a**2
    rhoavg = m / (4/3*pi*a**3)
    sigma  = lam[-1]+2*mu[-1]
    etabq  = lam[-1]+mu[-1]

    h_bq  = (m*g)/(4*pi*a**2*etabq) * (-sigma/mu[-1])
    l_bq  = (m*g)/(4*pi*a**2*etabq)
    k_bq  = (m*g)/(4*pi*a**2*etabq) * ( -3*rho[-1]*etabq / (2*rhoavg*mu[-1]) )

    return h_bq, l_bq, k_bq

def asymptotic_lln(r,rho,mu,lam,G=6.674e-11):
    """
    h_inf, l_inf, k_inf = asymptotic_lln(r,rho,mu,lam,G)

    Computes the asymptotic limit (for n->inf) of the LLNs for the given model,
    according to eqs (A51-A53) and (A83-A85) of Guo et al. (2004).

    Asymptotic expressions are given by

      h_n = h_inf[0] + h_inf[1]/n 
    n*l_n = l_inf[0] + l_inf[1]/n 
    n*k_n = k_inf[0] + k_inf[1]/n 
    
    :param r: Description
    :param rho: Description
    :param mu: Description
    :param lam: Description
    :param G: Description
    """

    a = r[-1]

    mass = 0.
    for i in range(len(r)):
        if i==0:
            r1 = 0.
        else:
            r1 = r[i-1]
        r2 = r[i]
        mass += 4/3 * pi * rho[i] * (r2**3 - r1**3)

    gR = G*mass/a**2

    rhoR = rho[-1]
    muR  = mu[-1]
    lamR = lam[-1]

    h1_inf = - gR**2 * (lamR + 2*muR) / (4 * pi * G * muR * (lamR + muR))
    l1_inf =   gR**2 / (4 * pi * G * (lamR + muR))
    k1_inf = - a * rhoR * gR / (2 * muR)

    h2_inf = gR**2 / (4 * pi * G * (lamR + muR)) * \
        ( -muR/(lamR + muR) + \
          a * rhoR * gR * (lamR**2 + lamR*muR - muR**2) / (2 * muR**2 * (lamR + muR)) + \
          2 * pi * G * a * rhoR * (lamR + muR) / (gR * muR) )

    l2_inf = gR**2 / (4 * pi * G * (lamR  + muR)) * \
        ( - (3*lamR**2 + 8*lamR*muR + 3*muR**2) / (2 * muR * (lamR + muR)) + \
          a * rhoR * gR * (lamR + 2*muR) / (2 * muR * (lamR + muR)))

    k2_inf = a * gR * rhoR / muR * \
        ( lamR / (4 * (lamR + muR)) + \
          a * rhoR * gR * (2*lamR + muR) / (8*muR*(lamR + muR)) + \
          pi * G * a * rhoR / gR )

    return np.array([h1_inf, h2_inf]), np.array([l1_inf, l2_inf]), np.array([k1_inf, k2_inf])