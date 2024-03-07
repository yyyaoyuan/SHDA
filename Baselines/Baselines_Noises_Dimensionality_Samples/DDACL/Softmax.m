function [proMatrix, preLabel] = Softmax(X,W,b,P)
%-----------------------------------------%
%Softmax classifier
% Input:
%       X: n * dx  % data matrix, n: the number of data, dx: the
%       dimension of the data
%       W: c * d   % parameter matrix, c: the number of labels, d: the
%       dimension of the subspace
%       b: c * 1   % parameter vector, c: the number of labels
%       P: dx * d  % projection matrix
% Output:
%       proMatrix: n * c
%       preLabel: n * 1
%-----------------------------------------%
[n,~] = size(X);
[c,~] = size(W);

% proMatrix = zeros(n,c);
% preLabel = zeros(n,1);
% for i = 1:n
%     proMatrix(i,:) = exp( W*P'*X(i,:)'+b )' / ( ones(1,c) * exp( W*P'*X(i,:)'+b ) );
%     [~,preLabel(i,1)] = max(proMatrix(i,:));
% end
% 
% proMatrix1 = (exp( X*P*W'+repmat(b,1,n)' )'./repmat(ones(1,c)*exp( X*P*W'+repmat(b,1,n)' )',c,1 ))';
% [~,preLabel1] = max(proMatrix1,[],2);

proMatrix = (exp( W*P'*X'+repmat(b,1,n) )./repmat(ones(1,c)*exp( W*P'*X'+repmat(b,1,n) ),c,1 ))';
[~,preLabel] = max(proMatrix,[],2);


end

