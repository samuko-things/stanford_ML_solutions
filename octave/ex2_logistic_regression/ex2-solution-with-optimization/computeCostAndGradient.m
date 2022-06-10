function [cost, grad] = computeCostAndGradient(theta, X, y, lambda)
% Initialize some useful values
m = length(y); % number of training examples
X = [ones(m,1) X]'; % add ones to X and transpose it
##X = X';
y = y';


% You need to return the following variables correctly 
cost = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h = sigmoid(theta'*X);

##cost = ( ((-1.*y)*log(h')) - ((1-y)*log(1-h')) )/m;
##
##grad = (X*(h-y)')./m;



cost = ( ((-1.*y)*log(h')) - ((1-y)*log(1-h')) )/m;
cost += ((theta(2:end)'*theta(2:end))*lambda)/(2*m);
##cost = cost + ((sum(theta(2:end).^2))*lambda)/(2*m);

grad = (X*(h-y)')./m;
grad(2:end) = grad(2:end) + (theta(2:end).*(lambda/m));

% =============================================================

end
