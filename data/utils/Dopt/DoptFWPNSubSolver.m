function [y, lambda, obj_val, info] = DoptFWPNSubSolver(x0, y0, A, At, gap_tol, max_iter)
% Conditional Gradient Method to solve the subproblem of D-optimal design:
%
%      min  <grad_f(x0), u> + 1/2< hessian_f(x0)*(u - x0), u - x0>
%      s.t. x in Simplex.
%
% lambda: (y - x0)'*hessian_f(x0)*(y - x0).

[n,m] = size(A);
Z = A*sparse(1:m,1:m,x0,m,m)*At;

% compute inverse matrix
R = chol(Z);
Rinv = inv(R);
InvZ = Rinv*Rinv';
InvZ = (InvZ+InvZ')/2;

% Calculate the objective value
obj_val = -sum(log(diag(R).^2));

% Get the Graient and Hessian
AtInvZA = At*InvZ*A;
Grad    = -diag(AtInvZA);
Hessian = AtInvZA.^2;

% Solve the subproblem
[y, info] = QuadSimplex_away_dopt(Grad - Hessian*x0, Hessian, y0, gap_tol, max_iter);
delta = y - x0;
lambda = sqrt(delta'*(Hessian*delta));

end