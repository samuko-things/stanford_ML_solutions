function J = computeCost1(X, y, theta)
% Initialize some useful values

m = length(y); % number of training examples
X_full = [ones(m, 1), X]; % Add a column of ones to X

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = sum(((X_full*theta')-y).^2)/(2*m);

% =========================================================================

end
