function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% h(x) for every row in X, given theta
pred = (sigmoid(theta' * X')');
% cost function
J = -1/m * ( y' * log(1 ./ (1 + exp(-(theta' * X')'))) + (1 - y)' * log( 1- (1 ./ (1 + exp(-(theta' * X')'))))) + lambda / (2*m) * sum(theta(2:length(theta)) .^ 2);

% gradient
grad = ((pred - y)' * X) / m;

% regularization (restoring theta(0) gradient)
temp = grad(1);
grad = grad + (lambda/m * theta)';
grad(1) = temp;

% =============================================================

end
