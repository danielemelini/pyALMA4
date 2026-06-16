% Computes the viscoelastic LNs for the M3-L70-V01 test suite model, invoking ISSM
% In the UM/LM we employ an EBM rheology

clear
nmax = 100;

% 1---- Initialize

md=model();
md.cluster=generic('name',oshostname(),'np',3);

md.materials=materials('litho');
md.miscellaneous.name='heavisideLNs_M3L70V01_EBM';

md.verbose=verbose('all');
md.verbose=verbose('1111111111111111');
yts=365.25*24*3600;

file_hypergeom = 'e:\work\ISSM\ISSM-Windows-MATLAB\test\Data\hypergeom.mat';

% 2---- Define rheological model

md.materials.numlayers=6;
md.materials.radius =  [10    1222.5   3480.0  5701.0   5951.0   6301.0   6371.0 ]'*1e3;
md.materials.density=  [     10750.0  10750.0  4978.0   3871.0   3438.0   3037.0 ]';
md.materials.lame_mu=  [        1e-5        0  2.2834   1.0549  0.70363  0.50605 ]'*1e11;
%md.materials.viscosity=[        20.0        0    20.0      0.5   0.0786    1e+25 ]'*1e21;
md.materials.viscosity=[           0        0       2        1        1    1e+25 ]'*1e21;
md.materials.lame_lambda=md.materials.lame_mu*0+5e17;
md.materials.issolid=[1 0 1 1 1 1]';
%md.materials.rheologymodel=zeros(md.materials.numlayers,1)+2;
%md.materials.rheologymodel(end)=0; %let the lithosphere be maxwell so we dont end up with a small unrelaxed mu for large timescales
md.materials.rheologymodel=[0 0 2 2 2 0]';
md.materials.ebm_alpha= ones(md.materials.numlayers,1)*.4;
md.materials.ebm_delta= ones(md.materials.numlayers,1)*.9;
%md.materials.ebm_taul= ones(md.materials.numlayers,1)*54*60;     %54min
%md.materials.ebm_tauh= ones(md.materials.numlayers,1)*7.134*yts; %7.134yr
md.materials.ebm_taul= ones(md.materials.numlayers,1)*(1/12)*yts; 
md.materials.ebm_tauh= ones(md.materials.numlayers,1)*     7*yts; 
%md.materials.setlitho2prem;
%lambda = md.materials.lame_lambda;
%md.materials.lame_lambda=md.materials.lame_mu*0+5e17;

load(file_hypergeom);
[nalpha, ~] = size(h1complex);
md.love.hypergeom_table1=h1real;
md.love.hypergeom_table2=h1real*0;
md.love.hypergeom_nalpha=101;
md.love.hypergeom_nz=length(z);
md.love.hypergeom_z=z;

% 3---- Set general parameters

md.love.frequencies=[0];
md.love.nfreq=length(md.love.frequencies);
md.love.underflow_tol=1e-20;
md.love.Gravitational_Constant=6.6732e-11;
md.love.min_integration_steps=100;

md.love.istemporal=1;
md.love.n_temporal_iterations=8;
%md.love.time=[0; (logspace(0,4.3, 24))'*yts];
%md.love.time=[0; (logspace(0,4.7, 30))'*yts];
%md.love.time=[0; (logspace(-1,4.7, 40))'*yts];
md.love.time=[0; (logspace(-3,4.7, 50))'*yts];
%md.love.time=[0; (logspace(0,5.3, 30))'*yts];
md.love=md.love.build_frequencies_from_time;

% 4---- Compute INCOMPRESSIBLE Love numbers

md.love.sh_nmin=1;
md.love.sh_nmax=nmax;
md.love.forcing_type=11;
md=solve(md,'lv');

n_load=1:nmax;
h_load=md.results.LoveSolution.LoveHt;
l_load=md.results.LoveSolution.LoveLt;
k_load=md.results.LoveSolution.LoveKt;

h_load=h_load(:,2:end);
l_load=l_load(:,2:end);
k_load=k_load(:,2:end);

% 7---- Define the compressible model

% Redefine the lambda vector
% according to the PREM averages computed by the
% build_compressible_M3L70V01 script

% Redefine the lambda vector
% according to the PREM averages computed by the
% build_compressible_M3L70V01 script

md.materials.lame_lambda=  [ 9.2629e+11 9.2629e+11 3.0172e+11 1.5559e+11 1.0038e+11 6.9170e+10]';

% 8---- Compute COMPRESSIBLE Love numbers

md.love.sh_nmin=1;
md.love.sh_nmax=nmax;
md.love.forcing_type=11;
md=solve(md,'lv');

hc_load=md.results.LoveSolution.LoveHt;
lc_load=md.results.LoveSolution.LoveLt;
kc_load=md.results.LoveSolution.LoveKt;

hc_load = hc_load(:,2:end);
lc_load = lc_load(:,2:end);
kc_load = kc_load(:,2:end);

%% 10---- Write output files

nt = length(md.love.time);

for j=1:6
    switch j

        case 1
            n = n_load; y = h_load;  f='h-load-heaviside-M3L70V01-EBM-incompressible.dat'; lab='h'; load='loading';
        case 2
            n = n_load; y = l_load;  f='l-load-heaviside-M3L70V01-EBM-incompressible.dat'; lab='l'; load='loading';
        case 3 
            n = n_load; y = k_load;  f='k-load-heaviside-M3L70V01-EBM-incompressible.dat'; lab='k'; load='loading';
        case 4
            n = n_load; y = hc_load; f='h-load-heaviside-M3L70V01-EBM-compressible.dat';   lab='h'; load='loading';
        case 5
            n = n_load; y = lc_load; f='l-load-heaviside-M3L70V01-EBM-compressible.dat';   lab='l'; load='loading';
        case 6 
            n = n_load; y = kc_load; f='k-load-heaviside-M3L70V01-EBM-compressible.dat';   lab='k'; load='loading';
    end

    fprintf( 'Writing file: %s\n', f );

    fmtstr = [ ' %4d' repmat( ' %14.6e', [1 nt] ) '\n' ];

    fid = fopen( f, 'w' );

    fprintf( fid, '# %s %s Love number for a Heaviside load\n', lab, load );
    fprintf( fid, '# Computed by ISSM on %s\n', datestr(now) );
    fprintf( fid, '#\n' );
    fprintf( fid, '#%4s %14s\n', 'n', [ lab '_n(t)' ] );

    for i=1:length(n)
        fprintf( fid, fmtstr, n(i), y(:,i) );
    end

    fid = fclose(fid);

end


