function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_row_no = length(theta);
D = zeros(theta_row_no,1);

for iter = 1:num_iters

##    % ====================== YOUR CODE HERE ======================
##    % Instructions: Perform a single gradient step on the parameter vector
##    %               theta. 
##    %
##    % Hint: While debugging, it can be useful to print out the values
##    %       of the cost function (computeCost) and gradient here.
##    %
##    
##    for r = 1:theta_row_no
##      x = X(:,r);
##      D(r,1) = (((X*theta)-y)' * x )/m
##    end
##    
##    theta = theta - (alpha*D)
##
##    % ============================================================
##
##    % Save the cost J in every iteration    
##    J_history(iter) = computeCost(X, y, theta);
    
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    [theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);


end

end