function CAE= est_entro_CAE(samp)
samp = cell2mat(samp)+1;
if isrow(samp)
    samp = samp.';
end
n = length(samp);
if n == 0
    CAE = 0;
    return
end
h1 = hist(samp,double(min(samp)):double(max(samp)));
fingerprint=hist(h1,0:max(h1));
fingerprint = fingerprint(2:end);
len_f = length(fingerprint);
p_s = 1 - fingerprint(1) / n;

prob = p_s * linspace(1/n,len_f/n,len_f);
if p_s == 0
    CAE = est_entro_MLE(samp);
else
    CAE = fingerprint *  (xlogx(prob) ./ (1 - (1 - prob).^n )');
    CAE = CAE / log(2);
end