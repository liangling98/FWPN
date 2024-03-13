function [x, w, y, info] = SimplexQPPAL(Q, q, l, u, options, x, w, y)
%% min_x 0.5<x, Qx> + <q,x> + delta_C(x) s.t. <e,x> = 1, C = {x| l <= x <= u}
%% max_{w,y,z} -0.5<w,Qw> + y - delta*_C(-z) s.t. -Qw + ye + z = q
tstart = tic;

if isfield(options, 'max_iter')
    max_iter = options.max_iter;
else
    max_iter = 100;
end
if isfield(options, 'max_iter_sub')
    max_iter_sub = options.max_iter_sub;
else
    max_iter_sub = 50;
end
if isfield(options, 'max_iter_step')
    max_iter_step = options.max_iter_step;
else
    max_iter_step = 20;
end
if isfield(options, 'max_iter_lin')
    max_iter_lin = options.max_iter_lin;
else
    max_iter_lin = 1000;
end
if isfield(options, 'stop_tol')
    stop_tol = options.stop_tol;
else
    stop_tol = 1e-8;
end
if isfield(options, 'stop_tol_sub')
    stop_tol_sub = options.stop_tol_sub;
else
    stop_tol_sub = 1e-9;
end
if isfield(options, 'stop_tol_lin')
    stop_tol_lin = options.stop_tol_lin;
else
    stop_tol_lin = 1e-10;
end
if isfield(options, 'sigma')
    sigma = options.sigma;
else
    sigma = 1.0;
end

n = length(q);

if nargin < 6
    x = zeros(n, 1);
    w = zeros(n, 1);
    y = 0;
    z = zeros(n, 1);
end

proj_C = @(x) min(u, max(l, x));

en = ones(n,1);
normq = norm(q);

Qx = Q*x;
Qw = Q*w;
Rd0 = Qw - y*en + q;
Rd = Rd0 - z;
Rp = sum(x) - 1;

errRp = abs(Rp);
errRd = norm(Rd) / max(1, normq);
errRq = norm(Qw - Qx) / max([1, norm(Qx), norm(Qw)]);
errRc = norm(x - proj_C(x - z)) / max([1, norm(x), norm(z)]);
pobj = 0.5 * sum(x .* Qx) + sum(q .* x);
dobj = -0.5 * sum(w .* Qw) + y;
relgap = abs(pobj - dobj) / max([1, pobj, dobj]);

info.errRp(1) = errRp;
info.errRd(1) = errRd;
info.errRq(1) = errRq;
info.errRc(1) = errRc;
info.pobj(1) = pobj;
info.dobj(1) = dobj;
info.relgap(1) = relgap;

for iter = 1:max_iter
    printf("\n %4d | %2.1e %2.1e %2.1e %2.1e | %-9.8e %- 9.8e %2.1e | %2.1e %2.1e |", ...
        iter, errRp, errRd, errRq, errRc, pobj, dobj, relgap, sigma, toc(tstart));

    if max([errRp, errRd, errRc, errRq, relgap]) < stop_tol
        break;
    end
    
    x_old = x;
    for iter_sub = 1:max_iter_sub
        x = proj_C(x - sigma*Rd0);
        Qx = Q*x;
        grad_Lw = Qw - Qx;
        grad_Ly = sum(x) - 1; 
        
    end

end


end