#!/usr/bin/env python3
#
#
from alma4 import *
alma_version='4.01'
alma_date='14-Sep-2026'
#
#
#
##########################################################################################################

import argparse
import sys
import datetime
from pathlib import Path

def write_output(n, t, ln, fname, ln_name, head_txt, labels):

    loadtype, run_type, model_file, comp_txt, outfmt, timeunits, timestamp = labels

    ndeg, nt = ln.shape

    f = open( fname, 'w' )

    f.write( '# ' + 58*'-' + '\n' )
    f.write( '# ' + run_type + ' analysis for the ' + ln_name + ' ' + loadtype + ' Love number\n' )
    f.write( '# Model name: ' + model_file + ' (' + comp_txt + ')\n' )
    if outfmt=='ln_vs_t':
        f.write( '# Times are in units of ' + timeunits + '\n' )
    f.write( '# File created by pyALMA4 on ' + timestamp + '\n' )
    f.write( '# ' + 58*'-' + '\n' )

    f.write( '#' + head_txt + '\n' )
        
    if outfmt=='ln_vs_n':
        for i in range(ndeg):
            f.write( ' {:6d}'.format( n[i] ))
            for k in range(nt):
                f.write( ' {:14.6e}'.format(ln[i,k]))
            f.write('\n')
    if outfmt=='ln_vs_t':
        for k in range(nt):
            f.write( ' {:14.6e}'.format( t[k] ))
            for i in range(len(n)):
                f.write( ' {:14.6e}'.format(ln[i,k]))
            f.write('\n')

    f.close()


def main():

    # Starting time

    t_start = time.perf_counter()

    # Set up the parser object and parse cmdline args

    parser = argparse.ArgumentParser( prog='alma4.py', \
             description='plAnetary Love nuMbers cAlculator', \
             epilog='DM, Sep 2026' )
    
    parser.add_argument( '--degrees',  type=str, required=True,  help='Harmonic degrees' )
    parser.add_argument( '--model',    type=str, required=True,  help='Path to the model file' )
    parser.add_argument( '--analysis', required=True, help='Analysis type',
                        choices=('elastic', 'heaviside', 'timedomain', 'frequency', 'laplace', \
                                 'fluid', 'fluidlimit', 'collocation') )
    parser.add_argument( '--incompressible', action='store_true', help='Assume incompressible rheology')
    parser.add_argument( '--log', action='store_true', help='Write a log file')
    load_group = parser.add_mutually_exclusive_group(required=True)
    load_group.add_argument('--loading', action='store_true', help='Compute loading LNs')
    load_group.add_argument('--tidal',   action='store_true', help='Compute tidal LNs')
    parser.add_argument( '--timesteps', type=str, help='Time step start, increment and end' )
    parser.add_argument( '--timescale', choices=('log', 'linear'), type=str, help='Time step scale' )
    parser.add_argument( '--label',     type=str, help='Label for output files' )
    parser.add_argument( '--timeunits', choices=('kyr', 'yr', 'days', 'hr'), default='kyr', help='Units for time-steps' )
    parser.add_argument( '--output-format', choices=('ln_vs_n', 'ln_vs_t'), default='ln_vs_n', help='Output format' )
    parser.add_argument( '--propagation', choices=('analytical', 'numerical','default'), default='default', help='Propagation method' )
    parser.add_argument( '--inversion', choices=('postwidder', 'talbot', 'default'), default='default', help='Laplace inversion method' )
    args = parser.parse_args()

    degrees     = args.degrees
    analysis    = args.analysis
    model_file  = args.model
    label       = args.label
    flag_inc    = args.incompressible
    flag_load   = args.loading
    flag_tide   = args.tidal
    timesteps   = args.timesteps
    timescale   = args.timescale
    timeunits   = args.timeunits
    outfmt      = args.output_format
    flag_log    = args.log
    propagation = args.propagation
    inversion   = args.inversion

    # Print a banner

    print( '' )
    print( ' *** This is pyALMA')
    print( ' *** (the plAnetary Love nuMbers cAlculator)')
    print( ' *** version ' + alma_version + ' - DM, ' + alma_date )
    print( '' )

    # Open the logfile, if requested

    if label is None:
        file_log = 'alma.log'
    else:
        file_log = 'alma-' + label + '.log'

    if file_log:
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        with open(file_log, 'w') as f:
            f.write( 'pyALMA job started on ' + timestamp + '\n' )

    # Define the list of harmonic degrees

    data = degrees.split(',')
    deg = []

    for i in range(len(data)):

        rng=data[i].split(':')

        if len(rng)==1:
            deg.append( np.array([int(rng[0])]) )
        elif len(rng)==2:
            nmin=int(rng[0])
            nmax=int(rng[1])
            deg.append( np.arange(nmin,nmax+1) )
        elif len(rng)==3:
            nmin=int(rng[0])
            nmax=int(rng[1])
            nstp=int(rng[2])
            deg.append( np.arange(nmin,nmax+1,nstp) )
        else:
            print( ' - ERROR: Invalid harmonic degree range "' + data[i] + '"')
            sys.exit(1)

    n = np.concatenate(deg)
    n = np.sort(n)

    # Define the forcing type and check if the requested LNs are consistent

    if flag_load:
        loadtype='loading'
        if np.any(n<1):
            print( ' - ERROR: loading LNs can be computed only for n>=1.')
            sys.exit(1)
    if flag_tide:
        loadtype='tidal'
        if np.any(n<2):
            print( ' - Error: tidal LNs can be computed only for n>=2.')
            sys.exit(1)

    # Check that the analysis type is valid

    print( ' - Analysis type: '+analysis)

    if file_log:
        with open(file_log, 'a') as f:
            f.write( 'Analysis type: ' + analysis + '\n' )
            if loadtype=='loading':
                f.write( 'Requested Love numbers are of the LOADING type\n' );
            elif loadtype=='tidal':
                f.write( 'Requested Love numbers are of the TIDAL type\n' );
            f.write('Love numbers will be computed for {} degrees between n={} and n={}\n'.format( \
                len(n), np.min(n), np.max(n) ))

    # Read the model file

    if file_log:
        with open(file_log, 'a') as f:
            f.write( 'Reading model file: ' + model_file + '\n' )

    if not Path(model_file).is_file():
        print( ' - ERROR: Cannot read model file '+model_file)
        sys.exit(1)

    print( ' - Reading model definition from file: '+model_file)

    data = np.genfromtxt( model_file, delimiter='\n', dtype=str )

    # Drop the header lines
    
    data = [x for x in data if not x.startswith('#')]

    # Build the model vectors

    nla = len(data)

    print( ' - The model contains {} layers'.format(nla) )

    r   = np.zeros( nla )
    rho = np.zeros( nla )
    mu  = np.zeros( nla )
    lam = np.zeros( nla )
    eta = np.zeros( nla )
    rheology = []
    rpar     = np.zeros((nla,5))

    for i in range(nla):
        idx = nla-i-1
        line = data[idx].split()

        r[i]     = float(line[1])
        rho[i]   = float(line[2])
        mu[i]    = float(line[3])
        lam[i]   = float(line[4])
        eta[i]   = float(line[5])

        rheology.append( line[6].lower() )
        if rheology[i] not in [ 'elastic', 'fluid', 'maxwell', 'newton', 'kelvin', \
                               'burgers', 'andrade', 'sundberg', 'sundcoop', 'ebm' ]:
            print( ' - Error: invalid rheology "' + rheology[i] + '"' )
            sys.exit(1)

        if rheology[i] not in [ 'elastic', 'fluid', 'maxwell', 'kelvin', 'newton' ]:
            if len(line)<6:
                print( ' - ERROR: missing rheological parameters for layer #', i )
                sys.exit(1)
            rheol_data = line[7:]

            if rheology[i]=='burgers':
                if len(rheol_data)<2:
                    print( ' - ERROR: incomplete rheological info for layer #', i )
                    print( '   For the Burgers rheology, provide mu2/mu and eta2/eta.')
                    sys.exit(1)            
                rpar[i,0] = float( rheol_data[0] )   # mu2/mu
                rpar[i,1] = float( rheol_data[1] )   # eta2/eta

            elif rheology[i]=='andrade':
                if len(rheol_data)<2:
                    print( ' - ERROR: incomplete rheological info for layer #', i )
                    print( '   For the Andrade rheology, provide alpha and zeta.')
                    sys.exit(1)            
                rpar[i,0] = float( rheol_data[1] )  # alpha
                rpar[i,1] = float( rheol_data[2] )  # zeta                     

            elif rheology[i] in [ 'sundberg', 'sundcoop' ]:
                if len(rheol_data)<4:
                    print( ' - ERROR: incomplete rheological info for layer #', i )
                    print( '   For the Sundberg-Cooper rheology, provide mu2/mu, eta2/eta, alpha and zeta.')
                    sys.exit(1)            
                rpar[i,0] = float( rheol_data[0] )  # mu2/mu
                rpar[i,1] = float( rheol_data[1] )  # eta2/eta                     
                rpar[i,2] = float( rheol_data[2] )  # alpha
                rpar[i,3] = float( rheol_data[3] )  # zeta                     

            elif rheology[i]=='ebm':
                if len(rheol_data)<4:
                    print( ' - ERROR: incomplete rheological info for layer #', i )
                    print( '   For the EBM rheology, provide alpha, delta, tauL and tauH.')
                    sys.exit(1)            
                rpar[i,0] = float( rheol_data[0] )  # alpha
                rpar[i,1] = float( rheol_data[1] )  # delta                     
                rpar[i,2] = float( rheol_data[2] )  # tauL
                rpar[i,3] = float( rheol_data[3] )  # tauH                     

    if flag_inc:
        lam=None
        print( ' - Model is incompressible' )
    else:
        print( ' - Model is compressible' )

    if file_log:
        with open(file_log, 'a') as f:
            f.write( 'Model file contains {} layers\n'.format(nla) )
            if flag_inc:
                f.write( 'INCOMPRESSIBLE rheology is assumed\n' )
            else:
                f.write( 'COMPRESSIBLE rheology is assumed\n' )
            f.write('\n')
            f.write( 'Rheological profile for the model (from surface to core):\n' )
            f.write( '  Lyr    r_top(m) rho(kg/m3)     mu(Pa)    lam(Pa)\n')
            for i in range(nla-1,-1,-1):
                f.write( ' {:4d}  {:10.4e} {:10.4e} {:10.4e}'.format(i,r[i],rho[i],mu[i]))
                if flag_inc:
                    f.write( ' {:>10s}'.format ('inf')   )
                else:
                    f.write( ' {:10.4e}'.format(lam[i] ))
                f.write(' '+rheology[i])
                if rheology[i] not in [ 'elastic', 'fluid' ]:
                    f.write( ', eta={:10.4e} Pa*s'.format(eta[i]) )
                if rheology[i]=='burgers':
                    f.write( ', mu2/mu={:10.4e}, eta2/eta={:10.4e}'.format(rpar[i,0], rpar[i,1]) )
                if rheology[i]=='andrade':
                    f.write( ', alpha={:10.4e}, zeta={:10.4e}'.format(rpar[i,0], rpar[i,1]) )
                if rheology[i] in [ 'sundcoop', 'sundberg' ]:
                    f.write( ', mu2/mu={:10.4e}, eta2/eta={:10.4e}, alpha={:10.4e}, zeta={:10.4e}'.format(
                        rpar[i,0], rpar[i,1], rpar[i,2], rpar[i,3]) )
                if rheology[i]=='ebm':
                    f.write( ', alpha={:10.4e}, delta={:10.4e}, tauL={:10.4e}{:s}, tauH={:10.4e}{:s}'.format(
                        rpar[i,0], rpar[i,1], rpar[i,2], timeunits, rpar[i,3], timeunits ) )
                    
                f.write('\n')

         
    # Build the time steps

    if analysis in ['elastic','fluid','fluidlimit'] and timesteps is not None:
        print( ' - Ignoring the timesteps definition for analysis type \'' + analysis + '\'' )
        timesteps = None

    if timesteps is None:

        t = 0.

        if analysis not in ['elastic','fluid','fluidlimit']:
            print(' - ERROR: timesteps are required for a '+analysis+' analysis')
            sys.exit(1)

    else:

        print( ' - Defining the time steps' )

        data = timesteps.split(',')

        t = []

        for i in range(len(data)):

            rng=data[i].split(':')

            if len(rng)==1:
                t.append( np.array([float(rng[0])]) )
            elif len(rng)==3:
                tmin=float(rng[0])
                tmax=float(rng[1])
                tstp=float(rng[2])
                if tstp>0:
                    t.append( np.arange(tmin,tmax+tstp/2,tstp) )
                elif tstp<0:
                    tstp=int(-tstp)
                    t.append( np.linspace(tmin,tmax,tstp) )
                else:
                    print( ' - ERROR: Invalid time step range "' + data[i] +'"')
                    sys.exit(1)            
            else:
                print( ' - ERROR: Invalid time step range "' + data[i] +'"')
                sys.exit(1)

        t = np.concatenate(t)
        t = np.sort(t)
          
        if timescale=='log':
            t = 10**t

        if np.any(t<=0):
            print( ' - ERROR: time steps must be >0')
            sys.exit(1)
            
        print( ' - Number of time steps: {}'.format(len(t)) )

    if np.isscalar(t):
        nt = 1
    else:
        nt = len(t)

    if timesteps is not None:
        if flag_log:
            with open(file_log,'a') as f:
                f.write('\n')
                f.write('Time steps:\n')
                for i in range(nt):
                    f.write('{:6d} {:12.6e}{:s}\n'.format( \
                        i+1, t[i], timeunits))

    # Set the propagation method

    if propagation=='analytical':
        numint=False
    elif propagation=='numerical':
        numint=True
    elif propagation=='default':
        numint=None

    if file_log:
        with open(file_log,'a') as f:

            f.write('\n')

            if numint is None:
                f.write( 'The default propagation scheme will be adopted\n' )
            elif numint==True:
                f.write( 'Numerical propagation has been selected\n' )
            elif numint==False:
                f.write( 'Analytical propagation has been selected\n' )

            if inversion=='default':
                f.write( 'The default Laplace inversion scheme will be invoked\n')
            elif inversion=='talbot':
                f.write( 'The Fixed-Talbot Laplace inversion scheme will be invoked\n')
            elif inversion=='postwidder':
                f.write( 'The Post-Widder Laplace inversion scheme will be invoked\n')

    # Call ALMA

    verbose=True
    ln_out = love_numbers(r,rho,mu,lam,eta,rheology,rpar,n,t,loadtype,analysis,verbose,numint=numint,inversion=inversion,timeunits=timeunits)

    if analysis=='collocation':
        he, le, ke, hv, lv, kv = ln_out
    else:
        hLN,lLN,kLN = ln_out
        
    # Define the output files

    if label is None:
        file_h, file_l, file_k = 'h.dat', 'l.dat', 'k.dat'
    else:
        file_h = 'h-' + label + '.dat'
        file_l = 'l-' + label + '.dat'
        file_k = 'k-' + label + '.dat'

    # Write log file, if requested

    #if flag_log:
    #
    #    print( ' - Writing log file: ' + file_log )
    #
    #    with open( file_log, 'w' ) as f:
    #
    #        f.write( 'Analysis type: {}\n'.format( analysis ) )
    #        f.write( 'Forcing type: {}\n'.format( loadtype ) )
    #        f.write( 'Model file: {}\n'.format(model_file) )
    #        f.write( 'Number of layers in model: {}\n'.format(nla) )
    #        f.write( 'Rheological profile for the model:\n' )
    #        f.write( '  Lyr    r_top(m) rho(kg/m3)     mu(Pa)    lam(Pa)\n')
    #        for i in range(nla-1,-1,-1):
    #            f.write( ' {:4d}  {:10.4e} {:10.4e} {:10.4e}'.format(i,r[i],rho[i],mu[i]))
    #            if flag_inc:
    #                f.write( ' {:>10s}'.format ('inf')   )
    #            else:
    #                f.write( ' {:10.4e}'.format(lam[i] ))
    #            f.write(' {:15s}'.format(rheology[i]))
    #            if rheology[i] not in [ 'elastic', 'fluid' ]:
    #                f.write( ' eta={:10.4e} Pa*s'.format(eta[i]) )
    #            f.write('\n')

    # Write LNs to output files

    if analysis in ['frequency', 'periodic']:
        re_im = '[re|im]_'
    else:
        re_im = ''

    print( ' - Writing output files: {:s}, {:s}, {:s}'.format( \
        re_im + file_h, re_im + file_l, re_im + file_k) ) 
    #print( '   h Love numbers: ' + re_im + file_h )
    #print( '   l Love numbers: ' + re_im + file_l )
    #print( '   k Love numbers: ' + re_im + file_k )
    
    # For elastic, fluid and fluid-limit analyses we override the output format
    # and reshape the LN vectors in order to use the same output routines used for
    # the time-dependent cases

    if analysis in [ 'elastic', 'fluid', 'fluidlimit' ]:
        if outfmt=='ln_vs_t':
            print( ' - WARNING: Ignoring the \'ln_vs_t\' output format' )
            outfmt='ln_vs_n'
        hLN = np.array( [hLN] ).T
        lLN = np.array( [lLN] ).T
        kLN = np.array( [kLN] ).T
    elif analysis=='collocation':
        if outfmt=='ln_vs_t':
            print( ' - WARNING: Ignoring the \'ln_vs_t\' output format' )
            outfmt='ln_vs_n'
        hLN = np.c_[ he, hv ]
        lLN = np.c_[ le, lv ]
        kLN = np.c_[ ke, kv ]
 
    if analysis=='elastic':
        run_type = 'Elastic'
    elif analysis=='fluid':
        run_type = 'Fluid'
    elif analysis=='fluidlimit':
        run_type = 'Fluid limit'
    elif analysis=='laplace':
        run_type = 'Laplace-domain'
    elif analysis in [ 'frequency', 'periodic' ]:
        run_type = 'Frequency-domain'
    elif analysis in ['heaviside', 'timedomain' ]:
        run_type = 'Time-domain (Heaviside)'
    elif analysis=='collocation':
        run_type = 'Collocation'

    if flag_inc:
        comp_txt = 'incompressible'
    else:
        comp_txt = 'compressible'

    # Define the file header

    if analysis in ('elastic'):
        h_head = ' Each row contains: n, hE_n'
        l_head = ' Each row contains: n, lE_n'
        k_head = ' Each row contains: n, kE_n'
    elif analysis in ('fluid', 'fluidlimit'):
        h_head = ' Each row contains: n, hF_n'
        l_head = ' Each row contains: n, lF_n'
        k_head = ' Each row contains: n, kF_n'
    elif analysis in [ 'heaviside', 'timedomain' ]:
        if outfmt=='ln_vs_n':
            h_head = ' Each row contains: n, hH_n(t1), ..., hH_n(tN)'
            l_head = ' Each row contains: n, lH_n(t1), ..., lH_n(tN)'
            k_head = ' Each row contains: n, kH_n(t1), ..., kH_n(tN)'
        if outfmt=='ln_vs_t':
            h_head = ' Each row contains: t ('+timeunits+'), h_n1(t), ..., h_nN(t)'
            l_head = ' Each row contains: t ('+timeunits+'), l_n1(t), ..., l_nN(t)'
            k_head = ' Each row contains: t ('+timeunits+'), k_n1(t), ..., k_nN(t)'
    elif analysis in [ 'frequency', 'periodic' ]:
        #h_head = '# T ('+timeunits+'), h_n(T)\n'
        #l_head = '# T ('+timeunits+'), l_n(T)\n'
        #k_head = '# T ('+timeunits+'), k_n(T)\n'
        if outfmt=='ln_vs_n':
            re_h_head = ' Each row contains: n, Re(h_n(T1)), ..., Re(h_n(TN))'
            re_l_head = ' Each row contains: n, Re(l_n(T1)), ..., Re(l_n(TN))'
            re_k_head = ' Each row contains: n, Re(k_n(T1)), ..., Re(k_n(TN))'
            im_h_head = ' Each row contains: n, Im(h_n(T1)), ..., Im(h_n(TN))'
            im_l_head = ' Each row contains: n, Im(l_n(T1)), ..., Im(l_n(TN))'
            im_k_head = ' Each row contains: n, Im(k_n(T1)), ..., Im(k_n(TN))'
        if outfmt=='ln_vs_t':
            re_h_head = ' Each row contains: T ('+timeunits+'), Re(h_n1(T)), ..., Re(h_nN(T))'
            re_l_head = ' Each row contains: T ('+timeunits+'), Re(l_n1(T)), ..., Re(l_nN(T))'
            re_k_head = ' Each row contains: T ('+timeunits+'), Re(k_n1(T)), ..., Re(k_nN(T))'
            im_h_head = ' Each row contains: T ('+timeunits+'), Im(h_n1(T)), ..., Im(h_nN(T))'
            im_l_head = ' Each row contains: T ('+timeunits+'), Im(l_n1(T)), ..., Im(l_nN(T))'
            im_k_head = ' Each row contains: T ('+timeunits+'), Im(k_n1(T)), ..., Im(k_nN(T))'
    elif analysis in [ 'laplace' ]:
        if outfmt=='ln_vs_n':
            h_head = ' Each row contains: n, h_n(s1), ..., h_n(sN)'
            l_head = ' Each row contains: n, l_n(s1), ..., l_n(sN)'
            k_head = ' Each row contains: n, k_n(s1), ..., k_n(sN)'
        if outfmt=='ln_vs_t':
            h_head = ' Each row contains: 1/s ('+timeunits+'), h_n1(s), ..., h_nN(s)'
            l_head = ' Each row contains: 1/s ('+timeunits+'), l_n1(s), ..., l_nN(s)'
            k_head = ' Each row contains: 1/s ('+timeunits+'), k_n1(s), ..., k_nN(s)'
    elif analysis in [ 'collocation' ]:
        h_head = ' Each row contains: n, hE_n, hV_n(s1), ..., hV_n(sN)'
        l_head = ' Each row contains: n, lE_n, lV_n(s1), ..., lV_n(sN)'
        k_head = ' Each row contains: n, kE_n, kV_n(s1), ..., kV_n(sN)'
    else:
        raise ValueError('Unexpected analysis type: '+analysis)

    timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')

    labels = loadtype, run_type, model_file, comp_txt, outfmt, timeunits, timestamp

    if analysis in [ 'elastic', 'fluid', 'fluidlimit', 'heaviside', 'timedomain', 'laplace', 'collocation' ]:
        write_output( n, t, hLN, file_h, 'h', h_head, labels )
        write_output( n, t, lLN, file_l, 'l', l_head, labels )
        write_output( n, t, kLN, file_k, 'k', k_head, labels )
    elif analysis in [ 'periodic', 'frequency' ]:
        write_output( n, t, np.real(hLN), 're_' + file_h, 'h', re_h_head, labels ) 
        write_output( n, t, np.real(lLN), 're_' + file_l, 'l', re_l_head, labels ) 
        write_output( n, t, np.real(kLN), 're_' + file_k, 'k', re_k_head, labels ) 
        write_output( n, t, np.imag(hLN), 'im_' + file_h, 'h', im_h_head, labels ) 
        write_output( n, t, np.imag(lLN), 'im_' + file_l, 'l', im_l_head, labels ) 
        write_output( n, t, np.imag(kLN), 'im_' + file_k, 'k', im_k_head, labels ) 

    """
    fh = open( file_h, 'w' )
    fl = open( file_l, 'w' )
    fk = open( file_k, 'w' )
                
    for (f,lab) in zip ( (fh,fl,fk), ('h','l','k') ):
        f.write( '# ' + 58*'-' + '\n' )
        f.write( '# ' + lab + ' ' + loadtype + ' Love number\n' )
        f.write( '# ' + run_type + ' analysis for model ' + model_file + '\n' )
        f.write( '# Model is ' + comp_txt + '\n' )
        if outfmt=='ln_vs_t':
            f.write( '# Times are in units of ' + timeunits + '\n' )
        f.write( '# File created by pyALMA4 on ' + timestamp + '\n' )
        f.write( '# ' + 58*'-' + '\n' )

    for (f,head) in zip ( (fh,fl,fk), (h_head, l_head, k_head)):
        f.write(head)
        
    for (f,v) in zip( (fh,fl,fk), (hLN,lLN,kLN) ):   
        if outfmt=='ln_vs_n':
            for i in range(len(n)):
                f.write( '{:6d}'.format( n[i] ))
                for k in range(nt):
                    f.write( ' {:14.6e}'.format(v[i,k]))
                f.write('\n')
        if outfmt=='ln_vs_t':
            for k in range(nt):
                f.write( '{:14.6e}'.format( t[k] ))
                for i in range(len(n)):
                    f.write( ' {:14.6e}'.format(v[i,k]))
                f.write('\n')

    for f in (fh,fl,fk):
        f.close()
    """

    if file_log:
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        with open(file_log, 'a') as f:
            f.write('\n')
            f.write( 'pyALMA job finished on ' + timestamp + '\n' )

    # Finish time

    t_end = time.perf_counter()

    print( ' - All done. Time elapsed: {:.3f} s'.format( t_end - t_start ) )
    print( '' )



if __name__ == "__main__":
    main()



