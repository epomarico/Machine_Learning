function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_test =  [0.01 0.05 0.1 0.5 1 5, 10 50];
sigma_test = [0.01 0.05 0.1 0.5 1 5 10 50];

C_min = C_test(1);
sigma_min = sigma_test(1);

error_min = Inf;

for i = 1:length(C_test)
    for j = 1:length(sigma_test)
        C_current = C_test(i);
        sigma_current = sigma_test(j);
        
        model = svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        
        if (error < error_min)
            error_min = error;
            C_min = C_current;
            sigma_min = sigma_current;
        end
    end
end

C = C_min
sigma = sigma_min




% =========================================================================

end
