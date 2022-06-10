
%% Initialization
clear ; close all; clc

%% ======================= Part 2: Plotting =======================

fprintf('Splitting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1:end-1); y = data(:, end);
n = size(X,2); % no of features

% Plot Data
% Note: You have to complete the code in plotData.m
fprintf('Plotting Data ...\n')
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;



%% =================== Part 3: Cost and Gradient descent ===================

theta_init = zeros(n+1, 1); % initialize fitting parameters

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
[J grad] = computeCostAndGradient(X, y, theta_init);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

% further testing of the cost function
[J grad] = computeCostAndGradient(X, y, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');
fprintf('Program paused. Press enter to continue.\n');
pause;








fprintf('\nRunning Gradient Descent ...\n')
% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

% run gradient descent
[theta, cost_history] = gradientDescent(X, y, theta_init, alpha, iterations);
fprintf('Cost at theta found by gradientDescent: %f\n', cost_history(end));
% print theta to screen
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');








fprintf('\nRunning Gradient Descent with optimized function - fminunc\n')
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', iterations);
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(computeCostAndGradient(X, y, t)), theta_init, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');








% Plot the linear fit
hold on; % keep previous plot visible
h = predict(X, theta);
plot(X, h, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure


% Predict values for population sizes of 35,000 and 70,000
h = predict(3.5, theta);
fprintf('For population = 35,000, we predict a profit of %f\n',...
    h*10000);
    
h = predict(7, theta);
fprintf('For population = 70,000, we predict a profit of %f\n',...
    h*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  [J_vals(i,j), _] = computeCostAndGradient(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
