function [x, info] = SimplexQPPAL(Q, q, l, u, options, x)
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
    max_iter_lin = 100;
end
if isfield(options, 'stop_tol')
    stop_tol = options.stop_tol;
else
    stop_tol = 1e-8;
end
if isfield(options, 'stop_tol_lin')
    stop_tol_lin = options.stop_tol_lin;
else
    stop_tol_lin = 1e-10;
end
if isfield(options, 'sigma')
    sigma = options.sigma;
else
    sigma = 1e0;
end
if isfield(options, 'lin_solver')
    lin_solver = options.lin_solver;
else
    lin_solver = 'bicgstab';
end
if isfield(options, 'verbose')
    verbose = options.verbose;
else
    verbose = false;
end

if ~isa(Q, "function_handle")
    Qmap = @(x) Q*x;
else
    Qmap = @(x) Q(x);
end

n = length(q);
proj_C = @(x) min(u, max(l, x));
en = ones(n,1);
normq = norm(q);

if nargin < 6
    x = ones(n, 1)/n;
end
w = x;
Qx = Qmap(x);
Qw = Qx;
y = sum(x .* Qx) + sum(q.*x);

z = q + Qw - y*en;
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

total_iter_lin = 0;
total_iter_sub = 0;

for iter = 1:max_iter
    if verbose
        fprintf("\n %4d | %2.1e %2.1e %2.1e %2.1e | %-9.8e %- 9.8e %2.1e | %2.1e %2.1e |", ...
            iter, errRp, errRd, errRq, errRc, pobj, dobj, relgap, sigma, toc(tstart));
    end

    if max([errRp, errRd, errRc, errRq, relgap]) < stop_tol
        break;
    end
        
    iter_ls = 0;
    in_prod = 0;
    res_lin = 0;
    iter_lin = 0;
    lin_solved = 0;

    x_old = x;
    x_input = x_old - sigma*Rd0;
    x = proj_C(x_input);
    Qx = Qmap(x);
    Ly = -sum(Rd0 .* x) - norm(x - x_old)^2 / (2 * sigma) + 0.5 * sum(w .* Qw) - y;

    stop_tol_sub = max(1e-8, 1e-1*errRp);
    for iter_sub = 1:max_iter_sub
        
        grad_Lw = Qw - Qx;
        grad_Ly = sum(x) - 1;   
        grad_L = [grad_Lw; grad_Ly];
        norm_grad = norm(grad_L);
        norm_Qx = norm(Qx);
        norm_Qw = norm(Qw);

        err_sub = norm_grad / max([1.0, norm_Qx, norm_Qw]);
        if verbose
            fprintf(" \n %8d | %- 9.8e | %2.1e %2.1e %2.1e | %4d %2.1e %d| %2d %- 2.1e |", ...
                iter_sub, Ly, err_sub, norm(grad_Lw), abs(grad_Ly),...
                iter_lin, res_lin, lin_solved, iter_ls, in_prod);
        end

        if err_sub < stop_tol_sub
            break;
        end

        uu = zeros(n,1);
        uu(x_input > l & x_input < u) = 1;       
        
        rhs = -[w-x; grad_Ly];
        if strcmp(lin_solver, 'bicgstab') || ~ismatrix(Q)
            V = @(x) Vmap(x, uu, Qmap, sigma);
            [d_wy, lin_solved, res_lin, iter_lin] = bicgstab(V, rhs, stop_tol_lin, max_iter_lin);
            iter_lin = floor(iter_lin);
            total_iter_lin = total_iter_lin + iter_lin;
        elseif strcmp(lin_solver, 'direct')
            sig_U = sigma*diag(uu);
            sig_UQ = sig_U * Q;
            V = [[eye(n) + sig_UQ, -sigma*uu]; [-sum(sig_UQ, 1), sigma*sum(uu)]];
            d_wy = V\rhs;
            res_lin = norm(V*d_wy - rhs) / (1+norm(rhs));
        end
        dw = d_wy(1:end-1);
        dy = d_wy(end);
        Qdw = Qmap(dw);
        in_prod = sum(d_wy .* grad_L);
        
        w_old = w;
        Qw_old = Qw;
        y_old = y;
        Ly_old = Ly;
        for iter_ls = 1:max_iter_step
            step = 0.5^(iter_ls-1);
            w = w_old + step*dw;
            y = y_old + step*dy;
            Qw = Qw_old + step*Qdw;
            Rd0 = Qw - y*en + q;
            x_input = x_old - sigma*Rd0;
            x = proj_C(x_input);
            
            Ly = -sum(Rd0 .* x) - norm(x - x_old)^2 / (2 * sigma) + 0.5 * sum(w .* Qw) - y;

            if (Ly - Ly_old) / max(1, Ly_old) < 1e-4*step*in_prod
                break;
            end
        end
        Qx = Qmap(x);
    end

    total_iter_sub = total_iter_sub + iter_sub;
    
    z = (x - x_input) / sigma;
    Rd = Rd0 - z;
    Rp = grad_Ly;
    errRp = abs(Rp);
    errRd = norm(Rd) / max(1, normq);
    errRq = norm(grad_Lw) / max([1, norm_Qx, norm_Qw]);
    errRc = norm(x - proj_C(x - z)) / max([1, norm(x), norm(z)]);
    pobj = 0.5 * sum(x .* Qx) + sum(q .* x);
    dobj = -0.5 * sum(w .* Qw) + y;
    relgap = abs(pobj - dobj) / max([1, pobj, dobj]);

    info.errRp(1+iter) = errRp;
    info.errRd(1+iter) = errRd;
    info.errRq(1+iter) = errRq;
    info.errRc(1+iter) = errRc;
    info.pobj(1+iter) = pobj;
    info.dobj(1+iter) = dobj;
    info.relgap(1+iter) = relgap;

    sigma = min(1e6, max(1e-2, sigma*2.5));
end
fprintf("\n");
info.iter = iter;
info.cputime = toc(tstart);
info.total_iter_lin = total_iter_lin;
info.total_iter_sub = total_iter_sub;
end

function Vx = Vmap(x, uu, Qmap, sigma)
ww = x(1:end-1);
yy = x(end);
Qww = Qmap(ww);
Vx = [ww + sigma*(uu.*Qww) - sigma * yy * uu; -sigma*sum(uu .* Qww) + sigma*yy*sum(uu)];
end