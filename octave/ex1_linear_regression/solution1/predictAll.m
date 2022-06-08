function h = predictAll(X, theta)
% Initialize some useful values

m = length(X(:,1)); % number of training examples
X_full = [ones(m, 1), X]; % Add a column of ones to X

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X_full*theta';

% =========================================================================

end
