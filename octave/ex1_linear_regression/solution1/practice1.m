
%% Initialization
clear ; close all; clc




%% ======================= Part 2: Plotting =======================
fprintf('Loading Data ...\n')
data = load('ex1data1.txt');
[X,y,n,m] = dataSplit(data);

% Plot Data
% Note: You have to complete the code in plotData.m
##plotData(X, y);





%% =================== Part 3: Cost and Gradient descent ===================


theta = zeros(1, n+1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost1(X, y, theta);
fprintf('With theta = [0 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

% further testing of the cost function
J = computeCost1(X, y, [-1 2]);
fprintf('\nWith theta = [-1 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');

fprintf('Program paused. Press enter to continue.\n');
pause;





fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
[theta, _] = gradientDescent1(X, y, theta, alpha, iterations);
fprintf('Theta found by gradient descent:\n');
theta

fprintf('Expected theta values (approx)\n\n');
fprintf(' -3.6303  1.1664\n');

t = normalEqn1(X,y);
fprintf('Theta found by normal equation:\n');
t

h_X = predictAll(X,theta);

% Plot the linear fit
plot(X, y, 'ro');
hold on; % keep previous plot visible
plot(X, h_X, 'g-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
h1 = predict(3.5,theta);
fprintf('For population = 35,000, we predict a profit of %f\n', h1*10000);
    
h2 = predict(7,theta);
fprintf('For population = 70,000, we predict a profit of %f\n', h2*10000);

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
	  t = [theta0_vals(i) theta1_vals(j)];
	  J_vals(i,j) = computeCost1(X, y, t);
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