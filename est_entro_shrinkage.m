function shrink = est_entro_shrinkage(samp,S)
samp = cell2mat(samp);
if isrow(samp)
    samp = samp.';
end
if nargin < 2
    S = length(unique(samp));
end
n = length(samp);
h1 = hist(samp,double(min(samp)):double(max(samp)));
fingerprint=hist(h1,0:max(h1));
fingerprint = fingerprint(2:end);
len_f = length(fingerprint);
prob = linspace(1/n,len_f/n,len_f);

MLE_squared = fingerprint * (prob.^2)';
lambda = (1 - MLE_squared) / (MLE_squared - 1/S) / (n-1);
shrink = (S - sum(fingerprint)) * xlogx(lambda/S) + fingerprint * xlogx(lambda/S + (1-lambda) * prob);
shrink = shrink / log(2);