function X_norm = featureNormalize1(X)

X_norm = X;
X_min = zeros(size(X, 2), 1);
X_max = zeros(size(X, 2), 1);


for count = 1:length(X_min)
  X_min(count) = min(X(:,count));
  X_max(count) = max(X(:,count));
endfor

for count = 1:length(X_min)
  X_norm(:,count) = (X_norm(:,count)-X_min(count))./X_max(count);
endfor

% ============================================================

end
