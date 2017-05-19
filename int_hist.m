function h = int_hist(x)
% INT_HIST(x) is a histogram of all integer values 1:max(x)

wid_x = size(x,2);
large = max(max(x));
h = zeros(large,wid_x);
for iter = 1:wid_x
    temp = full(sum(sparse(1:length(x(:,iter)),x(:,iter),1),1));
    h(1:length(temp),iter) = temp;
end

