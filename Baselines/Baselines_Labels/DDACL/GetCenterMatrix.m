function [Ms,Mt] = GetCenterMatrix(Xs, Xt, Xs_Label, Xl_Label, pseudo_Xu_Label)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[~,ds] = size(Xs);
[~,dt] = size(Xt);
Xt_Label = [Xl_Label;pseudo_Xu_Label];
Class = unique(Xs_Label);
c = length(Class);
Ms = zeros(c,ds);
Mt = zeros(c,dt);
for i = 1:c
   idxS = Xs_Label == Class(i);
   idxT = Xt_Label == Class(i);
   Ms(i,:) = mean(Xs(idxS,:))';
   Mt(i,:) = mean(Xt(idxT,:))';
end
end

