function [A_hat, E_hat, iter, err] = nrpca(D, opts)%lambda, tol, maxIter, mu_t, rho)

% April 2017
% This matlab code implements the Lp-RPCA method described in
% http://doi.org/10.1016/j.cviu.2017.03.002
%
% D - m x n matrix of observations/data (required input)
%
% opts - params for the algorithm including:
% opts.lambda - weight on sparse error term in the cost function
%
% opts.tol - tolerance for stopping criterion.
%          - DEFAULT 1e-7 if omitted.
%
% opts.maxit - maximum number of iterations
%            - DEFAULT 1000, if omitted.
%
% opts.rho   - step for mu rho > 1
%            - DEFAULT 1.5, if omitted.
%
% opts.mu_t  - mu multiplier
%            - DEFAULT 1.25, if omitted.
%
% opts.p     - p-value in Lp
%            - DEFAULT 0.6, if omitted.
%
%
% Copyright: Kha Gia Quach (k_q@encs.concordia.ca), April 2017
%  Concordia University

addpath PROPACK;

err = [];

[m n] = size(D);

if ~isfield(opts, 'lambda')
    lambda = 1 / sqrt(m);
else
    lambda = opts.lambda;
end

if ~isfield(opts, 'tol')
    tol = 1e-7;
else
    tol = opts.tol;
    if tol == -1
        tol = 1e-7;
    end
end

if ~isfield(opts, 'maxit')
    maxIter = 1000;
else
    maxIter = opts.maxit;
    if maxIter == -1
        maxIter = 1000;
    end
end

if ~isfield(opts, 'mu_t')
    mu_t = 1.25;
else
    mu_t = opts.mu_t;
end

if ~isfield(opts, 'rho')
    rho = 1.5;
else
    rho = opts.rho;
end

if ~isfield(opts, 'p')
    p = 0.6;
else
    p = opts.p;
end

% initialize
Y = D;
norm_two = lansvd(Y, 1, 'L');
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

A_hat = zeros( m, n);
E_hat = zeros( m, n);

A_old = zeros(m, n);
E_old = zeros(m, n);

mu = mu_t / norm_two;
mu_bar = mu * 1e7;
d_norm = norm(D, 'fro');

w = ones(m, n);
v = ones(min(m, n), 1);
epsilon = 1e-4;

iter = 0;
total_svd = 0;
converged = false;
stopCriterion = 1;
sv = 10;

while ~converged
    iter = iter + 1;
    
    Ymu = (1/mu) * Y;
    
    t = lambda / mu;

    temp_T = D - A_hat + Ymu;
    
    %% Solve l1 minimization for S
    E_hat = shrink_p(temp_T, t * w, p); % soft p-thresholding (lp)

    %% Update the weights
    w = p * (abs(E_hat) + epsilon) .^ (p - 1);
    
    temp_X = D - E_hat + Ymu;
    
    if choosvd(n, sv) == 1
        [U, S, V] = lansvd(temp_X, sv, 'L');
    else
        [U, S, V] = svd(temp_X, 'econ');
    end

    diagS = diag(S);
    
    % for lp norm
    svp = length(find(diagS > v(1:length(diagS)) ./ mu));
    newDiagS = shrink_p(diagS(1:length(diagS)), v(1:length(diagS)) ./ mu, p);      % for lp norm

    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    %% Solve l1 minimization for S
    A_hat = U(:, 1:svp) * diag(newDiagS(1:svp)) * V(:, 1:svp)';

    eps2 = eps(newDiagS);
    v(1:length(newDiagS)) = p * (newDiagS + eps2) .^ (p - 1);
    
    A_old = A_hat;
    E_old = E_hat;

    total_svd = total_svd + 1;
    
    Z = D - A_hat - E_hat;
    
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
    
    %% stop Criterion
    stopCriterion = norm(Z, 'fro') / d_norm;
    
    err = [err stopCriterion];
    
    if stopCriterion < tol
        converged = true;
    end
    
    if mod( total_svd, 10) == 0
        disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' stopCriterion ' num2str(stopCriterion)]);
    end
    
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;
    end
    
end
