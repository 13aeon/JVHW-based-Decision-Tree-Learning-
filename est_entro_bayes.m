function bayes = est_entro_bayes(samp,S,beta)
samp = cell2mat(samp);
if isrow(samp)
    samp = samp.';
end
if nargin < 2
    S = length(unique(samp));
    beta = sqrt(length(samp)) / S;
end
n = length(samp);
h1 = hist(samp,double(min(samp)):double(max(samp)));
fingerprint=hist(h1,0:max(h1));
fingerprint = fingerprint(2:end);
len_f = length(fingerprint);
fingerprint = [S - sum(fingerprint), fingerprint];
freq = (0:len_f);

bayes = - fingerprint * ( (freq+beta)/(n+S*beta) .* ( psi(freq + beta + 1) - psi(n+S*beta+1) ) )';
bayes = bayes / log(2);