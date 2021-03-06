##function [J grad] = nnCostFunction
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
##size(nn_params)    
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
##size(Theta1)
##size(Theta1_grad)
##size(Theta2)
##size(Theta2_grad)
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% calculating cost [J]

##a1 = X';
##a1 = [ones(m,1), a1];
##
##a2 = zeros(m, hidden_layer_size);
##a2 = sigmoid(a1*Theta1');
##a2 = [ones(m,1), a2];
##
##a3 = sigmoid(a2*Theta2');
##h = a3;


a1 = X';
a1 = [ones(1,m); a1];
a2 = sigmoid(Theta1*a1);
a2 = [ones(1,m); a2];
a3 = sigmoid(Theta2*a2);
h = a3;

Y = zeros(num_labels,m);
for count = 1:m
  Y(y(count)', count) = 1;    
endfor
##size(Y)

##J = ( ((-1.*y')*log(h)) - ((1-y)'*log(1-h)) )/m;
j1 = 0;
for i = 1:m
  j1 += (Y(:,i)'*log(h(:,i))) + ((1-Y(:,i))'*log(1-h(:,i)));    
endfor

j21 = sum(Theta1(:,2:end).^2);
j21 = sum(j21);

j22 = sum(Theta2(:,2:end).^2);
j22 = sum(j22);

j2 = j21+j22;

J = (j1/-m)+((j2*lambda)/(2*m));




% calculating the gradient [grad]

Delta1 = zeros(size(Theta1_grad));
##size(Delta1)
Delta2 = zeros(size(Theta2_grad));


d3 = a3-Y;
d2 = (Theta2'*d3) .* (a2.*(1-a2));
d2 = d2(2:end,:);

Delta1 = d2*a1';
Delta2 = d3*a2';

##for i = 1:m
##  a_1 = X(i,:)';
##  a_1 = [1; a_1];
##  a_2 = sigmoid(Theta1*a_1);
##  a_2 = [1; a_2];
##  a_3 = sigmoid(Theta2*a_2);
##  
##  
##  d3 = a_3-Y(:,i);
##  d2 = (Theta2'*d3) .* (a_2.*(1-a_2));
##  d2 = d2(2:end);
##  
##  Delta1 = Delta1 + (d2*a_1');
##  Delta2 = Delta2 + (d3*a_2');
##endfor

% =========================================================================

% Unroll gradients
##Theta1_grad = Delta1./m;
##Theta2_grad = Delta2./m;


Theta1_reg = [zeros(size(Theta1,1),1) ((Theta1(:,2:end).*lambda)./m)];
Theta2_reg = [zeros(size(Theta2,1),1) ((Theta2(:,2:end).*lambda)./m)];

##Theta1_reg = [Theta1(:,1) ((Theta1(:,2:end).*lambda)./m)];
##Theta2_reg = [Theta2(:,1) ((Theta2(:,2:end).*lambda)./m)];

Theta1_grad = (Delta1./m) + Theta1_reg;
Theta2_grad = (Delta2./m) + Theta2_reg;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
##size(grad)

end
