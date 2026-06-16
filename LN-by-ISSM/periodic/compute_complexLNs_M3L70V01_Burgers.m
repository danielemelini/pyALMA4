% Computes the complex tidal LNs of degree 2 for the M3-L70-V01 model

clear

% 1---- Initialize

md=model();
md.cluster=generic('name',oshostname(),'np',3);

md.materials=materials('litho');
md.miscellaneous.name='complexLNs_M3L70V01_Burgers';

md.verbose=verbose('all');
yts=365.25*24*3600;

% 2---- Define rheological model

md.materials.numlayers=6;
md.materials.radius =  [10    1222.5   3480.0  5701.0   5951.0   6301.0   6371.0 ]'*1e3;
md.materials.density=  [     10750.0  10750.0  4978.0   3871.0   3438.0   3037.0 ]';
md.materials.lame_mu=  [        1e-5        0  2.2834   1.0549  0.70363  0.50605 ]'*1e11;
md.materials.viscosity=[           0        0       2        1        1    1e+25 ]'*1e21;
md.materials.lame_lambda=md.materials.lame_mu*0+5e17;
md.materials.issolid=[1 0 1 1 1 1]';
md.materials.rheologymodel=[0 0 1 1 1 0]';
md.materials.burgers_mu   =md.materials.lame_mu * 0.1;
md.materials.burgers_viscosity=md.materials.viscosity * 0.01;

% 3---- Set general parameters

%md.love.frequencies=[0 logspace(-8,3,100)/yts];
md.love.frequencies=[0 logspace(-8,3,200)/yts];
%md.love.frequencies=[0 logspace(-8,5,150)/yts];
md.love.nfreq=length(md.love.frequencies);
md.love.sh_nmin=2;
md.love.sh_nmax=2;
md.love.underflow_tol=1e-20;
md.love.pw_threshold=1e-3;
md.love.Gravitational_Constant=6.6732e-11;
md.love.min_integration_steps=500;
md.love.max_integration_dr=5e3;
md.love.complex_computation=1;

md.love.istemporal=0;
md.love.time=[];

md.love.love_kernels=0;

% 4---- Compute LNs

md.love.forcing_type=9;
md=solve(md,'lv');

hr=md.results.LoveSolution.LoveHf(:,3);
hi=md.results.LoveSolution.LoveHfi(:,3);

lr=md.results.LoveSolution.LoveLf(:,3);
li=md.results.LoveSolution.LoveLfi(:,3);

kr=md.results.LoveSolution.LoveKf(:,3);
ki=md.results.LoveSolution.LoveKfi(:,3);

% 5---- Define the compressible model

% Redefine the lambda vector
% according to the PREM averages computed by the
% build_compressible_M3L70V01 script

md.materials.lame_lambda=  [ 9.2629e+11 9.2629e+11 3.0172e+11 1.5559e+11 1.0038e+11 6.9170e+10]';

% 6---- Compute LNs

md.love.forcing_type=9;
md=solve(md,'lv');

hcr=md.results.LoveSolution.LoveHf(:,3);
hci=md.results.LoveSolution.LoveHfi(:,3);

lcr=md.results.LoveSolution.LoveLf(:,3);
lci=md.results.LoveSolution.LoveLfi(:,3);

kcr=md.results.LoveSolution.LoveKf(:,3);
kci=md.results.LoveSolution.LoveKfi(:,3);

% 7---- Write output files

t  = (1 ./ md.love.frequencies) / yts;       % Period

nt = length(md.love.frequencies);

for j=1:2
    
    switch j
        case 1
            f = 'tln-complex-M3L70V01-Burgers-incompressible.dat'; 
            re_h = hr;  im_h = hi;
            re_l = lr;  im_l = li;
            re_k = kr;  im_k = ki;
        case 2
            f = 'tln-complex-M3L70V01-Burgers-compressible.dat'; 
            re_h = hcr; im_h = hci;
            re_l = lcr; im_l = lci;
            re_k = kcr; im_k = kci;
    end

    fprintf( 'Writing file: %s\n', f );

    fmtstr = [ repmat( ' %14.6e', [1 7] ) '\n' ];

    fid = fopen( f, 'w' );

    fprintf( fid, '# Complex Love numbers for a periodic load\n' );
    fprintf( fid, '# Computed by ISSM on %s\n', datestr(now) );
    fprintf( fid, '#\n' );
    fprintf( fid, '#%14s %14s %14s %14s %14s %14s %14s\n', ...
        'T (yr)', 'Re(h_2)', 'Im(h_2)', ...
                  'Re(l_2)', 'Im(l_2)', ...
                  'Re(k_2)', 'Im(k_2)' );

    for i=1:nt
        fprintf( fid, fmtstr, ...
            t(i), ...
            re_h(i), im_h(i), ...
            re_l(i), im_l(i), ...
            re_k(i), im_k(i) );
    end

    fid = fclose(fid);

end

