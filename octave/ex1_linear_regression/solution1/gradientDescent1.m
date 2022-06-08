function [theta, J_history] = gradientDescent1(X, y, theta, alpha, num_iters)
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_row_no = length(theta);

gradient = zeros(1, theta_row_no);

X_full = [ones(m, 1), X]; % Add a column of ones to x
for iter = 1:num_iters
    
    gradient = (((X_full*theta')-y)'*X_full)/m  ;
    
    theta = theta - (alpha*gradient);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost1(X, y, theta);

end

end