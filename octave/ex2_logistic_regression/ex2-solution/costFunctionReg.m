##function [J, grad] = costFunctionReg(theta, X, y, lambda)
##%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
##%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
##%   theta as the parameter for regularized logistic regression and the
##%   gradient of the cost w.r.t. to the parameters. 
##
##% Initialize some useful values
##m = length(y); % number of training examples
##
##% You need to return the following variables correctly 
##J = 0;
##grad = zeros(size(theta));
##
##% ====================== YOUR CODE HERE ======================
##% Instructions: Compute the cost of a particular choice of theta.
##%               You should set J to the cost.
##%               Compute the partial derivatives and set grad to the partial
##%               derivatives of the cost w.r.t. each parameter in theta
##
##
##h = sigmoid(X*theta);
##
##J = ( ((-1.*y')*log(h)) - ((1-y)'*log(1-h)) )/m;
##J = J+ ((theta(2:end)'*theta(2:end))*lambda)/(2*m);
##
##grad = (X'*(h-y))./m;
##grad(2:end) = grad(2:end) + (theta(2:end).*(lambda/m));
##
##
##
##% =============================================================
##
##end




function [cost, grad] = costFunctionReg(theta, X, y, lambda)
% Initialize some useful values
m = length(y); % number of training examples
##X = [ones(m,1) X]'; % add ones to X and transpose it
X = X';
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
