function BUB = est_entro_BUB(samp,S)
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
[val_vec,~] = BUBfunc(n,S,11,0);
BUB = [S-sum(fingerprint),fingerprint] *  val_vec(1:len_f+1);
BUB = BUB / log(2);