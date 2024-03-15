function hist = DoptPNQPPALSolver(A, x, options, tols)
time1 = tic;
m = size(A, 2);
% A_scale = norm(A, 'fro')^2;
At = A';
Z = A*sparse(1:m,1:m,x,m,m)*At;

R = chol(Z);
Rinv = inv(R);
InvZ = Rinv*Rinv';
InvZ = (InvZ+InvZ')/2;
AtInvZA  = At*InvZ*A;
Grad = -diag(AtInvZA);

l = zeros(m, 1);
u = 1e20*ones(m,1);

for iter = 1:options.Miter
    Hess =  AtInvZA.^2;
    Hess_scale = norm(Hess, 'fro');

    Q = Hess / Hess_scale;
    q = Grad / Hess_scale - Q*x;

    par.stop_tol = 1e-10;
    par.max_iter = 50;
    par.lin_solver = 'bicgstab';
    par.max_iter_lin = 20;
    par.verbose = false;
    [z, info] = SimplexQPPAL(Q, q, l, u, par, x);

    dx = z - x;
    rdiff = norm(x - PortProxSplx(x - Grad)) / max([1.0, norm(x), norm(Grad)]);
    hist.err(iter, 1) = rdiff;
    step = 1.0;
    x = x + step * dx;

    Z = A*sparse(1:m,1:m,x,m,m)*At;

    R = chol(Z);
    Rinv = inv(R);
    InvZ = Rinv*Rinv';
    InvZ = (InvZ+InvZ')/2;
    AtInvZA  = At*InvZ*A;
    Grad = -diag(AtInvZA);

    hist.f(iter) = -sum(log(diag(R).^2));
    hist.cumul_time(iter) = toc(time1);

    fprintf(" %4d  | %- 9.8e | %3.2e | %3.2e | %2d %5d %5d \n", ...
        iter, hist.f(iter), rdiff, hist.cumul_time(iter), ...
        info.iter, info.total_iter_sub, info.total_iter_lin);

    if rdiff < tols.main && iter > 1
        hist.msg = "Convergence achieved!";
        break;
    end
end
if iter >= options.Miter
    hist.msg = 'Exceed the maximum number of iterations';
end
hist.time = toc(time1);
hist.iter = iter;
hist.xopt = x;
hist.obj = -sum(log(diag(R).^2));
end