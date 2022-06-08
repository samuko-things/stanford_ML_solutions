function [X_norm, mu, sigma] = featureNormalize2(X)

% You need to set these values correctly
X_norm = X;
mu = zeros(size(X, 2), 1);
sigma = zeros(size(X, 2), 1);


% compute the mean and std of the datas and update it
for count = 1:length(mu)
  mu(count) = mean(X(:,count));
  sigma(count) = std(X(:,count));
endfor

for count = 1:length(mu)
  X_norm(:,count) = (X_norm(:,count)-mu(count))./sigma(count);
endfor

% ============================================================

end
