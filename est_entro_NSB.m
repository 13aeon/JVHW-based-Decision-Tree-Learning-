function NSB = est_entro_NSB(samp,S)
samp = cell2mat(samp);
if isrow(samp)
    samp = samp.';
end
if nargin < 2
    S = length(unique(samp));
end
n = length(samp);
if n == 0
    NSB = 0;
    return
else
h1 = hist(samp,double(min(samp)):double(max(samp)));
fingerprint=hist(h1,0:max(h1));
fingerprint = fingerprint(2:end);
len_f = length(fingerprint);
freq = (1:len_f);

K1 = sum(fingerprint);
fingerprint = [S - K1, fingerprint];
if K1 == n || S == 1 || n == S
    NSB = est_entro_MLE(samp);
    return
end
% Compute optimal k0
vec_b = [(n-1)/(2*n), (1-2*n)/(3*n), (n^2-n-2)/9/(n^2-n)];
k0_ini = max(n * (1-K1/n).^(-1:1) * vec_b', 1e-4);
k0 = fzero(@(x) solve_k0(n,K1,x), k0_ini,optimset('Display','off'));
if ~isfinite(k0)
    k0 = k0_ini;
end

% Compute optimal k
k1 = fingerprint(2:end) * (psi(freq) - psi(1))' / (K1/k0^2 - psi(1,k0) + psi(1,k0+n));
k2 = (  (K1/k0^3 + (psi(2,k0)-psi(2,k0+n)) / 2) * k1^2 + k0 *  fingerprint(2:end) * (psi(1,freq) - psi(1,1))'  ) ...
    / (K1/k0^2 - psi(1,k0) + psi(1,k0+n));
k_ini = max(k0 + k1/S + k2/S^2, 1e-4);
k = fzero(@(x) solve_k(fingerprint,x), k_ini,optimset('Display','off'));
if ~isfinite(k)
    k = k_ini;
end

% Numerical Integration
val_max = fingerprint * gammaln( (0:len_f) + k/S )' - gammaln(n + k) + gammaln(k) - S*gammaln(k/S);
curvature = fingerprint * psi(1, (0:len_f) + k/S)' - S^2 * psi(1,n+k) + S^2 * psi(1,k) - S*psi(1,k/S);

D = 50;
denom = NaN; numer = NaN;

while ~isfinite(denom) || ~isfinite(numer)
    denom = quadgk(@(beta) p_density(fingerprint,beta,val_max) .* (S * psi(1,S*beta+1) - psi(1,beta+1)),...
        max(  0, k/S - D/sqrt(abs(curvature))  ) , k/S + D / sqrt(abs(curvature)));
    numer = quadgk(@(beta) p_density(fingerprint,beta,val_max) .* bayes_entro(fingerprint,beta) .* (S * psi(1,S*beta+1) - psi(1,beta+1)),...
        max(  0, k/S - D/sqrt(abs(curvature))  ) , k/S + D / sqrt(abs(curvature)));
    D = D/2;
end

NSB = numer / denom / log(2);
end
end

function res = solve_k0(n,K1,x)
    res = psi_m(x+n) - psi_m(x) - K1./x;
end

function res = solve_k(fingerprint,x)
    len_f = length(fingerprint);
    S = sum(fingerprint);
    freq = (0:len_f-1);
    n = fingerprint * freq';
    
    res = - psi_m(x/S) + psi_m(x) - psi_m(x+n);
    for iter = 1 : len_f
        res = res + fingerprint(iter) * psi_m(iter - 1 + x/S) / S ;
    end
end

function res = psi_m(x)
    res = - Inf * ones(size(x));
    res(x>0) = psi(x(x>0));
end

% function res = solve_beta(S,x,xi)
%     res = psi(S*x+1) - psi(x+1) - xi;
% end

function res = p_density(fingerprint,beta,val_max)
    len_f = length(fingerprint);
    S = sum(fingerprint);
    freq = (0:len_f-1);
    n = fingerprint * freq';

    %beta = fzero(@(x) solve_beta(S,x,xi),0);
    res_log = - gammaln(n + S*beta) + gammaln(S*beta) - S*gammaln(beta) - val_max;
    for iter = 1 : len_f
        res_log = res_log + fingerprint(iter) * gammaln(iter - 1 + beta);
    end
    res = exp(res_log);
end

function res = bayes_entro(fingerprint,beta)
    len_f = length(fingerprint);
    S = sum(fingerprint);
    freq = (0:len_f-1);
    n = fingerprint * freq';

    %beta = fzero(@(x) solve_beta(S,x,xi),0);
    res = 0;
    for iter = 1 : len_f
        res = res - fingerprint(iter) * ( (iter-1+beta)./(n+S*beta) .* ( psi(iter + beta) - psi(n+S*beta+1) ) );
    end
end