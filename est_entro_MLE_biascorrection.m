function MLE_MM = est_entro_MLE_biascorrection(samp)
samp = cell2mat(samp);
if isrow(samp)
    samp = samp.';
end
[n, wid] = size(samp);
if n == 1
    samp = samp'; 
    n = wid;
end

MLE_MM = est_entro_MLE(samp) + (length(unique(samp)) - 1) /2/n/log(2);