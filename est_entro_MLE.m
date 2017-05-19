function est = est_entro_MLE(samp)
% est = est_entro_MLE(samp) 
%
% This function returns our scalar estimate of the entropy (in bits) of samp when
% samp is a vector, and returns a row vector consisting of the entropy
% estimate of each column of samp when samp is a matrix. 
%
% Input: 
% ----- samp: a vector or matrix which can only contain integers
% Output: 
% ----- est: the entropy (in bits) of the input vector or that of each column of the
%            input matrix
    
    test_integer = abs(samp - fix(samp));
    if max(test_integer(:))>=1e-5
        error('Input sample must only contain integers!');
    end
    
    [n, wid] = size(samp);
    if n == 1
        samp = samp'; 
        n = wid;
        wid = 1;
    end

    %samp = samp - double(min(min(samp))) + 1;
    for iter = 1:wid
        [~, ~, ic] = unique(samp(:,iter));
        samp(:, iter) = ic;
    end
    h1 = int_hist(samp);
    fingerprint = int_hist(h1+1);
    [len_f,wid_f] = size(fingerprint);
    fingerprint = fingerprint(2:end,:);
    est = zeros(1,wid_f);
    
    if len_f>1
        prob = linspace(1/n,(len_f-1)/n,len_f-1);
        prob_mat = xlogx(prob);
        est = prob_mat * fingerprint;
    end
    
    est = est / log(2);
end

function output = xlogx(x)
    non_zero = find(x >= 1e-10);
    output = zeros(size(x));
    output(non_zero) = -x(non_zero) .* log(x(non_zero));
end

function h = int_hist(x)
% INT_HIST(x) is a histogram of all integer values 1:max(x)

wid_x = size(x,2);
large = max(max(x));
h = zeros(large,wid_x);
for iter = 1:wid_x
    temp = full(sum(sparse(1:length(x(:,iter)),x(:,iter),1),1));
    h(1:length(temp),iter) = temp;
end
end