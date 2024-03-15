function hist = PortPNQPPALSolver(W, x, options, tols)
n = size(W,1);
p = size(W,2);
hist.nsize = [n,p];

if ~isfield(options,'cpr')
    options.cpr = 0; % no comparison is made
end

use_cvx = 0;
time1 = tic;
denom = W * x;

l = zeros(p, 1);
u = 1e20*ones(p, 1);

W_scale = norm(W, 'fro')^2/1e1;
ratG = 1.0 ./ denom;
Grad = - W' * ratG;
for iter = 1:options.Miter
    
     
    Hess = @(x) (W' * ((W*x) .* (ratG.^2))) / W_scale;

    par.stop_tol = 1e-10;
    par.max_iter = 50;
    par.lin_solver = 'bicgstab';
    par.max_iter_lin = 20;
    par.verbose = false;
    q = Grad/W_scale - Hess(x);
    [z, info] = SimplexQPPAL(Hess, q, l, u, par, x);

    dx = z - x;
    %lambda = sqrt(sum(dx.*(Hess(dx))));
    %rdiff = lambda / max(1.0, norm(x));
    rdiff = norm(x - PortProxSplx(x - Grad)) / max([1.0, norm(x), norm(Grad)]);
    hist.err(iter, 1) = rdiff;
    %step = 1 / (1 + lambda);
    step = 1.0;
    x = x + step * dx;

    denom = W * x;
    ratG = 1.0 ./ denom;
    Grad = - W' * ratG;

    hist.f(iter) = -sum(log(denom));
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
hist.obj = -sum(log(denom));
end