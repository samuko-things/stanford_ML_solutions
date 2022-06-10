function [cost grad] = computeCostAndGradient(X, y, theta)
% Initialize some useful values
m = length(y); % number of training examples
X = [ones(m,1) X]'; % add ones to X and transpose it
y = y';


% You need to return the following variables correctly 
cost = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = theta'*X;

cost = sum( (h-y).^2 )/(2*m);

grad = (X*(h-y)')./m;

% =========================================================================

end
