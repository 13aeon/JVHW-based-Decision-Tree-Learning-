function dirichlet =est_entro_Dirichlet(samp,S,a)
samp = cell2mat(samp);
if isrow(samp)
    samp = samp.';
end
if nargin < 2
    S = length(unique(samp));
    a = sqrt(length(samp)) / S;
end
n = length(samp);
h1 = hist(samp,double(min(samp)):double(max(samp)));
fingerprint=hist(h1,0:max(h1));
fingerprint = fingerprint(2:end);
len_f = length(fingerprint);
prob = ( (1:len_f) + a ) / (n+S*a);
dirichlet = fingerprint *  xlogx(prob);
dirichlet = dirichlet ./ log(2);