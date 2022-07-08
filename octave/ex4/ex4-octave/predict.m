function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);


% You need to return the following variables correctly 
p = zeros(m, 1);


a1 = X';
a1 = [ones(1,m); a1];
a2 = sigmoid(Theta1*a1);
a2 = [ones(1,m); a2];
h = sigmoid(Theta2*a2);

[_, p] = max(h', [], 2);

% =========================================================================


end
