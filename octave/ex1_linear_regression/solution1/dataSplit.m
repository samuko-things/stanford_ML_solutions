function [X, y, n, m] = dataSplit(data)

data_width = length(data(1,:));
X = data(:, 1:data_width-1); y = data(:, data_width);
m = length(y); % number of training examples
n = length(X(1,:));

end