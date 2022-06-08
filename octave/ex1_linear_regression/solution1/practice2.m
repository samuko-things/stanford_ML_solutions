
%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc



fprintf('Loading Data ...\n')
data = load('ex1data2.txt');
[X,y,n,m] = dataSplit(data);

##% Print out some data points
##fprintf('First 10 examples from the dataset: \n');
##fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
##
##fprintf('Program paused. Press enter to continue.\n');
##pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

##X_norm = featureNormalize1(X);
[X_norm, _, _] = featureNormalize2(X);

%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(1, n+1); %n=no of features plus 1 for theta0
[theta, J_history] = gradientDescent1(X_norm, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
theta

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = predict([1650, 3],theta);


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;




%% ================ Part 3: Normal Equations ================

% Calculate the parameters from the normal equation
t = normalEqn1(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
t
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = predict([1650, 3],t);


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

