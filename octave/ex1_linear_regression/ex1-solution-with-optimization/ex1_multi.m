%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data1.txt');
X = data(:, 1:end-1); y = data(:, end);
n = size(X,2); % no of features



% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

X = featureNormalize(X);


%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.5;
iterations = 1500;

% Init Theta and Run Gradient Descent 
theta = zeros(n+1, 1);
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);
J_history(end)
% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', iterations);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(computeCostAndGradient(t, X, y)), theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('\n');




% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
##price = predict(X, theta); % You should change this
##
##% ============================================================
##
##fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
##         '(using gradient descent):\n $%f\n'], price);
##
##fprintf('Program paused. Press enter to continue.\n');
##pause;


%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:end-1); y = data(:, end);
n = size(X,2); % no of features

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


X_norm = featureNormalize(X);

% Choose some alpha value
alpha = 0.01;
num_iters = 1000;

% Init Theta and Run Gradient Descent 
theta = zeros(n+1, 1);

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', iterations);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(computeCostAndGradient(t, X, y)), theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('\n');



####89597.909542 
#### 139.210674 
#### -8738.019112
##
####% Estimate the price of a 1650 sq-ft, 3 br house
####% ====================== YOUR CODE HERE ======================
####price = 0; % You should change this
####
####
####% ============================================================
####
####fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
####         '(using normal equations):\n $%f\n'], price);
####
