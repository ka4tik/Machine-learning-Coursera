function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    subtheta0=0;
    for i=1:m,
       subtheta0+=(theta(1,1)+ theta(2,1)*X(i,2) - y(i,1));
    end;
    subtheta0=subtheta0/m;
    
    

    subtheta1=0;
    for i=1:m,
       subtheta1+=((theta(1,1)+ theta(2,1)*X(i,2) - y(i,1))*X(i,2));
    end;
    subtheta1=subtheta1/m;

    theta(1,1)=theta(1,1)- alpha*subtheta0;
    theta(2,1)=theta(2,1)- alpha*subtheta1;
    %disp(theta(1,1));
    %disp(theta(2,1));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);  
    %disp(J_history(iter));
end

end