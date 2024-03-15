clc;
clear all;
addpath(genpath(pwd));
warning off;

run_PN = 1;
run_FWPN = 1;
run_FWQPPAL = 1;
run_FWQP = 1;

%% Choosing the data
use_real_data = 0;
if use_real_data == 1
    id = 1;
    plist = {'473500_wk.mat','625723_wk.mat','625889_wk.mat'};
    pname = plist{id};
    load(pname);
    [n,p] = size(W);
    fprintf(' We are solving real problem of size n is %5d and p is %5d.\n', n,p);
    fprintf('\n');
else
    n = 1e+3;
    p = 5e+2;
    W = PortGenData(n, p, 0.1);
    fprintf(' We are solving synthetic problem of size n is %5d and p is %5d.\n', n,p);
    fprintf('\n');
end

%% Preprocessing the data    
if (min(min(W)) >= 0)
    fprintf('Data is valid.\n');
    fprintf('\n');
else
    fprintf('Data is adjusted by taking exp.\n');
    fprintf('\n');
    W = exp(W);
end
x0    = ones(p,1)/p;

%% Solving the problem by using standard Proximal Newton method
if run_PN == 1
    fprintf('************************************************************************\n')
    fprintf('*************** Solving portfolio problem by using PN ******************\n')
    fprintf('************************************************************************\n')
    options.Miter       = 20;
    options.printst     = 1;
    tols.main           = 1e-8;
    options.Lest        = 1;
    hist_PN = PortPNSolver(W, x0, options, tols);
end

%% Solving the problem by using our method
if run_FWPN == 1
    Options.lambda0    = 1;
    Options.lambda_tol = 1e-8;
    Options.sub_tol    = 0.1;
    Options.short2long = 10;
    Options.max_iter   = 20;
    Options.sub_max_iter = size(W,2)/5;
    get_obj   = @(x) -sum(log(W*x));
    SubSolver = @(x, y, tol, max_iter) PortFWPNSubSolver(x, y, W, tol, max_iter);
    fprintf('************************************************************************\n')
    fprintf('********** Solving portfolio problem by using FWPN *********************\n')
    fprintf('************************************************************************\n')
    hist_FWPN = ProxNSolver(W, x0, Options, SubSolver, get_obj);
end

%% Solving the problem by uisng FWQPPAL
if run_FWQPPAL == 1
    fprintf('************************************************************************\n')
    fprintf('********** Solving portfolio problem by using FWQPPAL ******************\n')
    fprintf('************************************************************************\n')
    options.Miter = 20;
    tols.main = 1e-8;
    hist_FWQPPAL = PortFWQPPALSolver(W, x0, options, tols);
end

%% Solving the problem by uisng FWQPPAL
if run_FWQP == 1
    fprintf('************************************************************************\n')
    fprintf('********** Solving portfolio problem by using FWQP *********************\n')
    fprintf('************************************************************************\n')
    options.Miter = 20;
    tols.main = 1e-8;
    hist_FWQP = PortFWQPSolver(W, x0, options, tols);
end

%% Plot 
semilogy(1:length(hist_FWQPPAL.err), hist_FWQPPAL.err, ...
    1:length(hist_FWPN.err), hist_FWPN.err, ...
    1:length(hist_PN.err), hist_PN.err, ...
    1:length(hist_FWQP.err), hist_FWQP.err);
legend("FWQPPAL", "FWPN", "PN", "FWQP");


