function Grass = est_entro_Grass(samp)
samp = cell2mat(samp);
if isrow(samp)
    samp = samp.';
end
n = length(samp);
h1 = hist(samp,double(min(samp)):double(max(samp)));
fingerprint=hist(h1,0:max(h1));
fingerprint = fingerprint(2:end);
len_f = length(fingerprint);

g_func = zeros(len_f, 1);
for i = 1 : len_f
    g_func(i) = psi(i) + (-1)^i * quadgk(@(x) x.^(i-1) ./ (1+x), 0, 1); 
end

Grass = log(n) -  fingerprint * ( (1:len_f)' .* g_func(1:len_f) ) / n;
Grass = Grass / log(2);