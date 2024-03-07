function [Wt,bt,Pt,Ps,VectorObj] = DDA(Xs,Xs_Label,Xl,Xl_Label,Xu,Xu_Label,beta,tau,lambda,d,T)
%-----------------------------------------%
%CTF algorithm
% Input:
%       Xs: ns * ds  % data matrix, ns: the number of source data, dx: the
%       dimension of source data
%       Xs_Label: ns * 1   % source data label
%       Xl: nl * dt  % data matrix, nl: the number of labeled target data, dt: the
%       dimension of target data
%       Xl_Label: nl * 1   % target labeled data label
%       Xu: nu * dt  % data matrix, nu: the number of unlabeled target data, dt: the
%       dimension of target data
%       Xu_Label: ni * 1   % target unlabeled data label
% Output:
%       Wt: c * d  % target classifier parameter matrix, c: the number of labels, d: the
%       dimension of the subspace
%       bt: c * 1  % target classifier parameter vector, c: the number of labels
%       Pt: dt * d % target projection matrix, dt: the dimension of target data, d: the dimension of the subspace
%-----------------------------------------%
[ns,ds] = size(Xs);
[nl,dt] = size(Xl);
[nu,~] = size(Xu);
nt = nl+nu;
c = length(unique(Xs_Label));  % the number of labels
Xt = [Xl;Xu]; 

Ys = zeros(ns,c);
Yl = zeros(nl,c);
%---------------------------------%
indexYs = sub2ind(size(Ys),(1:ns)',Xs_Label); % get index of Ys according to row and col
Ys(indexYs) = 1;
indexYl = sub2ind(size(Yl),(1:nl)',Xl_Label); % get index of Yl according to row and col
Yl(indexYl) = 1;
%---------------------------------%
Wt = normrnd(0,0.01,c,d);    % target classifier parameter matrix
bt = normrnd(0,0.01,c,1);    % target classifier parameter vector
Ps = normrnd(0,0.01,ds,d);   % source projection matrix
Pt = normrnd(0,0.01,dt,d);   % target projection matrix
%---------------------------------%
Para_ft = [Wt(:); bt; Pt(:)];   % put all parameters into an column vector
%---------------------------------%
% CTF algorithm
% pseudo_Xu_Label_ft = [];
[Wt, bt, Pt, ~] = Update_parameters_ft( Para_ft, Xl, Yl, lambda, d);
[~,pseudo_Xu_Label_ft] = Softmax(Xu,Wt,bt,Pt);   % ft(xu);

for i = 1:T
    %-----------------------%
    % MMD matrix
    Mk = ComputerMmdMatrix(Xs_Label, Xl_Label, pseudo_Xu_Label_ft, ns, nt, c);
    % class center matrix
    [Ms,Mt] = GetCenterMatrix(Xs, Xt, Xs_Label, Xl_Label, pseudo_Xu_Label_ft);
    % Laplacian matrix
    A = (ones(c,c)-eye(c))*(1/c^2);
    D = diag(sparse(sqrt(1./sum(A))));
    L = speye(c)-D*A*D;
    %-----------------------%   
    Para = [Wt(:); bt; Ps(:); Pt(:)];   % put all parameters into an column vector
    [Wt, bt, Ps, Pt, VectorObj] = ...
        Update_parameters( Para,  Xs, Ys, Xl, Yl, Xu, Mk, Ms, Mt, L, beta, tau, lambda, d);
    [~,pseudo_Xu_Label_ft] = Softmax(Xu,Wt,bt,Pt);   % ft(xu);
    [~,pseudo_Xl_Label_ft] = Softmax(Xl,Wt,bt,Pt);   % ft(xl);
    [~,pseudo_Xs_Label_ft] = Softmax(Xs,Wt,bt,Ps);   % ft(xs);
    acc_fu = Evaluate(pseudo_Xu_Label_ft,Xu_Label)*100;
    acc_fl = Evaluate(pseudo_Xl_Label_ft,Xl_Label)*100;
    acc_fs = Evaluate(pseudo_Xs_Label_ft,Xs_Label)*100;
    fprintf('ft(xu) accuracy is:%f\n',acc_fu);
    fprintf('ft(xl) accuracy is:%f\n',acc_fl);
    fprintf('ft(xs) accuracy is:%f\n',acc_fs);
    fprintf('--------------------------\n');
end



