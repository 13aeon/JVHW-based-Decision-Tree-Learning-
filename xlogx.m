function output = xlogx(x)
non_zero = find(x >= 1e-10);
output = zeros(length(x),1);
output(non_zero) = -x(non_zero) .* log(x(non_zero));
end