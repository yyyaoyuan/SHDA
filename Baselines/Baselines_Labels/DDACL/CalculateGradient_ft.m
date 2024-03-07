function [Obj, G_Para] = CalculateGradient_ft( Para, varargin)
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
Xl = varargin{1};
Yl = varargin{2};
lambda = varargin{3};
d = varargin{4};  % the dimension of the common space
%--------------------%
[nl,dt] = size(Xl);
[~,c] = size(Yl);  % the number of labels
%-----------------------------------------------------%
i_Wt = 0;   % the index of Wt
i_bt = i_Wt+c*d;   % the index of bt
i_Pt = i_bt+c*1;  % the index of Pt
% Get all parameters 
Wt = reshape(Para(i_Wt+1:i_Wt+c*d, 1), c, d);
bt = Para(i_bt+1:i_bt+c*1, 1);
Pt = reshape(Para(i_Pt+1:i_Pt+dt*d, 1), dt, d);
%----------------------------------------------------%
[Tl, ~] = Softmax(Xl,Wt,bt,Pt);  % Tl = ft(Xl)
%----------------------------------------------------%
Obj = - (1/nl)*trace(Yl*log(Tl')) ...
      + (lambda/2)*( norm(Wt,'fro')^2 + norm(Pt,'fro')^2 );

%  fprintf('MMD_same is: %f \n', (1/2)*M_same);
%  fprintf('D_same is: %f\n', (1/2)*D_same);
%  fprintf('Xl_Loss is: %f\n', -(1/nl)*trace(Yl*log(Tl')));
%  fprintf('Xs_Loss is: %f\n', -(1/ns)*trace(Ys*log(Ts')));
%--------------------------------------------------------------------------%
% Gradients
%----------------------------------------------------------------------%
% G_Wt
G_Wt = (1/nl)*(Tl-Yl)'*Xl*Pt + lambda*Wt;
% G_bt
G_bt = (1/nl)*(Tl-Yl)'*ones(nl,1);
% G_Pt
G_Pt = (1/nl)*Xl'*(Tl-Yl)*Wt + lambda*Pt;
%--------------------------------------------------------------------------%
G_Para = [G_Wt(:); G_bt;G_Pt(:)];
%--------------------------------------------------------------------------%
end

