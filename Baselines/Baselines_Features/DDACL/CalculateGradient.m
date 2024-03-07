function [Obj, G_Para] = CalculateGradient( Para, varargin)
%-----------------------------------------%
%CalculateGradient function
% Input:
%       Para: parameter vector, Para = [Ws(:); bs; Wt(:); bt; Ps(:); Pt(:)]
%       varargin: other parameters;
% Output:
%       Obj: the value of the loss function
%       G_Para: the gradient vector, G_Para = [G_Ws(:); G_bs; G_Wt(:); G_bt; G_Ps(:); G_Pt(:)]
%------------------%
% [Obj, G_Para] = CalculateGradient( Para, Xs, Ys, Xs_Label, Xl, Yl, Xl_Label, Xu, pseudo_Xu_Label, alpha, beta, lambda, d)
%-----------------------------------------------------%
% Get all constant matrix;
Xs = varargin{1};
Ys = varargin{2};
Xl = varargin{3};
Yl = varargin{4};
Xu = varargin{5};
Mk = varargin{6};
Ms = varargin{7};
Mt = varargin{8};
L = varargin{9};
beta = varargin{10};
tau = varargin{11};
lambda = varargin{12};
d = varargin{13};  % the dimension of the common space
%--------------------%
[ns,ds] = size(Xs);
[nl,dt] = size(Xl);
[nu,~] = size(Xu);
[~,c] = size(Ys);  % the number of labels
nt = nl+nu;
%-----------------------------------------------------%
i_Wt = 0;   % the index of Wt
i_bt = i_Wt+c*d;   % the index of bt
i_Ps = i_bt+c*1;   % the index of Ps
i_Pt = i_Ps+ds*d;  % the index of Pt
% Get all parameters 
Wt = reshape(Para(i_Wt+1:i_Wt+c*d, 1), c, d);
bt = Para(i_bt+1:i_bt+c*1, 1);
Ps = reshape(Para(i_Ps+1:i_Ps+ds*d, 1), ds, d);
Pt = reshape(Para(i_Pt+1:i_Pt+dt*d, 1), dt, d);
%----------------------------------------------------%
[Ts, ~] = Softmax(Xs,Wt,bt,Ps);  % Ts = ft(Xs)
[Tl, ~] = Softmax(Xl,Wt,bt,Pt);  % Tl = ft(Xl)
%----------------------------------------------------%
Xt = [Xl;Xu]; 
X = [Xs*Ps;Xt*Pt];
% MMD matrix
%Mk = ComputerMmdMatrix(Xs_Label, Xl_Label, pseudo_Xu_Label, ns, nt, c);
Mk1 = Mk(1:ns,1:ns);
Mk2 = Mk(1:ns,ns+1:ns+nt);
Mk3 = Mk(ns+1:ns+nt,1:ns);
Mk4 = Mk(ns+1:ns+nt,ns+1:ns+nt);
%-----------------------%
% class center matrix
%[Ms,Mt] = GetCenterMatrix(Xs, Xt, Xs_Label, Xl_Label, pseudo_Xu_Label);
% Laplacian matrix
%A = (ones(c,c)-eye(c))*(1/c^2);
%D = diag(sparse(sqrt(1./sum(A))));
%L = speye(c)-D*A*D;
%----------------------------------------------------%
% Total losses
M_same = trace(X'*Mk*X);
D_same = trace(Ps'*Ms'*L*Ms*Ps + Pt'*Mt'*L*Mt*Pt);

Obj = - (1/nl)*trace(Yl*log(Tl')) - (1/ns)*trace(Ys*log(Ts')) ...
      + (beta/2)*(M_same) - (tau/2)*(D_same) ...
      + (lambda/2)*( norm(Wt,'fro')^2 + norm(Ps,'fro')^2 + norm(Pt,'fro')^2 );
% Obj = (1/(2*nl))*norm((Yl-Tl),'fro')^2 + (1/(2*ns))*norm((Ys-Ts),'fro')^2 ...
%       + (beta/2)*(M_same) - (tau/2)*(D_same) ...
%       + (lambda/2)*( norm(Wt,'fro')^2 + norm(Ps,'fro')^2 + norm(Pt,'fro')^2 );
% fprintf('Obj is: %f \n', Obj);
%   fprintf('MMD_same is: %f \n', (1/2)*M_same);
%   fprintf('D_same is: %f\n', -(1/2)*D_same);
%   fprintf('Xl_Loss is: %f\n', -(1/nl)*trace(Yl*log(Tl')));
%   fprintf('Xs_Loss is: %f\n', -(1/ns)*trace(Ys*log(Ts')));
%   fprintf('---------------------------\n');
%--------------------------------------------------------------------------%
% Gradients
%------------------------------------------------------------------%
% temp variables
% D_Ps;
M_same_Ps = (Xs'*Mk1*Xs*Ps + Xs'*Mk2*Xt*Pt);
D_same_Ps = Ms'*L*Ms*Ps;
% D_Pt;
M_same_Pt = (Xt'*Mk4*Xt*Pt + Xt'*Mk3*Xs*Ps);
D_same_Pt = Mt'*L*Mt*Pt;
%----------------------------------------------------------------------%
% G_Wt
G_Wt = (1/nl)*(Tl-Yl)'*Xl*Pt + (1/ns)*(Ts-Ys)'*Xs*Ps + lambda*Wt;
% G_bt
G_bt = (1/nl)*(Tl-Yl)'*ones(nl,1) + (1/ns)*(Ts-Ys)'*ones(ns,1);
% G_Ps
G_Ps = (1/ns)*Xs'*(Ts-Ys)*Wt + beta*M_same_Ps - tau*D_same_Ps + lambda*Ps;
% G_Pt
G_Pt = (1/nl)*Xl'*(Tl-Yl)*Wt + beta*M_same_Pt - tau*D_same_Pt + lambda*Pt;
%--------------------------------------------------------------------------%
G_Para = [G_Wt(:); G_bt; G_Ps(:); G_Pt(:)];
%--------------------------------------------------------------------------%
end

