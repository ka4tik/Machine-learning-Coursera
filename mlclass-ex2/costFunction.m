function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%disp(m);
features=(size(theta))(1,1);
for i=1:m,
h=0;
for j=1:features,
h+=theta(j,1)*X(i,j);
end;
h=sigmoid(h);
J+=(-y(i,1)*log(h)-((1-y(i,1))*log(1-h)));
end;
J=J/(m);


sizegrad=(size(grad))(1,1);
%disp(sizegrad);
for i=1:sizegrad,
     for j=1:m,
      h=0;
      for k=1:sizegrad,
        h+=theta(k,1)*X(j,k);
      end;
      h=sigmoid(h);
      grad(i,1)+=(h-y(j,1))*X(j,i);
     end;
     grad(i,1)=grad(i,1)/m;
end;
     



% =============================================================

end
