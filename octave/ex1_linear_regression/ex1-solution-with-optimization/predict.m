function h = predict(X, theta)
% Initialize some useful values

m = size(X,1); % number of training examples
X = [ones(m, 1), X]'; % Add a column of ones to X

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = theta'*X;

h = h';
% =========================================================================

end