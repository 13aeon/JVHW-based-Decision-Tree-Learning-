function jack=est_entro_jackknife(samp)
samp = cell2mat(samp);
if isrow(samp)
    samp = samp.';
end
n = length(samp);
h1 = hist(samp,double(min(samp)):double(max(samp)));
fingerprint=hist(h1,0:max(h1));
fingerprint = fingerprint(2:end);
len_f = length(fingerprint);
prob = linspace(1/(n-1),len_f/(n-1),len_f);
jack = fingerprint * (  (1-prob)' .* xlogx(prob) + prob' .* xlogx(prob-1/n)  ) / log(2);
jack = est_entro_MLE(samp) * n - jack * (n-1);
