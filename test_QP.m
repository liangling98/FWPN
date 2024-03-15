clc;
clear;
addpath(genpath(pwd));

n = 1000;
Q = randn(n, n);
Q = Q*Q';
q = rand(n, 1);

[x_qp, w_qp, y_qp, info_qp] = SimplexQPPAL(Q, q, zeros(n,1), 1e20*ones(n,1), []);

tic;
cvx_solver gurobi
cvx_begin 
    cvx_precision high
    variable x(n)
    minimize (0.5*x' * Q * x + q' * x)
    sum(x) == 1
    x >= 0 
cvx_end
time_cvx = toc;