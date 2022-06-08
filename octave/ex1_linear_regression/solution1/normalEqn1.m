function [theta] = normalEqn1(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.
m = length(y);
X_full = [ones(m, 1), X]; % Add a column of ones to X
N = length(X_full(1,:));
theta = zeros(1, N);


theta = (pinv(X_full'*X_full)*X_full'*y)';


% ============================================================

end
