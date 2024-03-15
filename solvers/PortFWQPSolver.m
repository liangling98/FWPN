function hist = PortFWQPSolver(W, x, options, tols)
n = size(W,1);
p = size(W,2);
hist.nsize = [n,p];

if ~isfield(options,'cpr')
    options.cpr = 0; % no comparison is made
end

time1 = tic;
denom = W * x;

ratG = 1.0 ./ denom;
Grad = - W' * ratG;
for iter = 1:options.Miter
    
     
    Hess = W' * diag(ratG.^2) * W;

    cvx_solver mosek
    cvx_begin quiet
        cvx_precision high 
        variable z(p)
        minimize (Grad' * (z-x) + 0.5 * (z-x)' * (Hess * (z-x)))
        sum(z) == 1
        z >= 0
    cvx_end

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

    fprintf(" %4d  | %- 9.8e | %3.2e | %3.2e \n", ...
        iter, hist.f(iter), rdiff, hist.cumul_time(iter));

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