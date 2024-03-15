clc;
clear all;
addpath(genpath(pwd));

%% Generate Data
rng(123);
n = 1e2;
p = 5e3;

mu    = zeros(n,1);
sigma = rand(n,n);
sigma = sigma'*sigma;
A = mvnrnd(mu,sigma,p)';
At = A';

run_FWPN = 1;
run_PNQPPAL = 1;

if run_FWPN == 1
    fprintf('************************************************************************\n')
    fprintf('****** Solving D-optimal design problem by FWPN ************************\n')
    fprintf('************************************************************************\n')
    x0 = 1/p*ones(p,1);
    Options.lambda0 = 10;
    Options.lambda_tol = 1e-8;
    Options.sub_tol    = 0.01;
    Options.short2long = 10;
    Options.max_iter   = 100;
    Options.sub_max_iter = 10*size(A,2);
    get_obj   = @(x) DoptGetObj(x, A, At);
    SubSolver = @(x, y, tol, max_iter) DoptFWPNSubSolver(x, y, A, At, tol, max_iter);
    grad_map = @(x) DoptGetGrad(x, A, At);
    hist_FWPN = ProxNSolver(grad_map, x0, Options, SubSolver, get_obj);
end

%% Solving the problem by uisng FWQPPAL
if run_PNQPPAL == 1
    fprintf('************************************************************************\n')
    fprintf('********** Solving portfolio problem by using PNQPPAL ******************\n')
    fprintf('************************************************************************\n')
    options.Miter = 20;
    tols.main = 1e-8;
    hist_PNQPPAL = DoptPNQPPALSolver(A, x0, options, tols);
end

%% Plot 
semilogy(1:length(hist_PNQPPAL.err), hist_PNQPPAL.err, ...
    1:length(hist_FWPN.err), hist_FWPN.err);
legend("FWQPPAL", "FWPN", "PN");
