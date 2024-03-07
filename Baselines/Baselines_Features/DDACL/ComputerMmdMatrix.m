function M = ComputerMmdMatrix(Xs_Label, Xl_Label, pseudo_Xu_Label, ns, nt, c)
%COMPUTER_MMD_MATRIX Summary of this function goes here
%   Detailed explanation goes here
Xt_Label = [Xl_Label;pseudo_Xu_Label];
e0 = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e0*e0';
for i = reshape(unique(Xs_Label),1,c)
    e = zeros(ns + nt,1);
    e(Xs_Label == i) = 1/length(find(Xs_Label == i));
    e(ns+find(Xt_Label==i)) = -1/length(find(Xt_Label==i));
    e(isinf(e)) = 0;
    M = M + e*e';
end
M = M/norm(M,'fro');
end

