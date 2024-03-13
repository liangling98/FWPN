function hist = PortTRFWSolver(W, x, options, tols)
n = size(W,1);
p = size(W,2);
hist.nsize = [n,p];

if ~isfield(options,'cpr')
    options.cpr = 0; % no comparison is made
end

% alpha = 0.9/n^0.5;
% alpha = 0.1;

time1 = tic;
denom = W * x;

for iter = 1:options.Miter
    
    ratG = 1.0 ./ denom;
    Grad = - W' * ratG;    
    Hess = W' * diag(ratG.^2) * W;

    alpha = 0.9;
    
    cvx_solver mosek
    cvx_begin quiet
        cvx_precision high
        variable z(p)
        minimize (sum((Grad - Hess*x) .* z) + 0.5 * z' * Hess * z)
        sum(z) == 1.0 
        z >= 0.0
    cvx_end 

    dx = z - x;
    rdiff = abs(sum(Grad .* dx)) / max(1.0, norm(x));
    hist.err(iter, 1) = rdiff;
    x = z;
    denom = W * x;
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