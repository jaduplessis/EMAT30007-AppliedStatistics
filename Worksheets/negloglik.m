%% function that returns the negative log likelihood of a simple linear model given parameter values and data

function [out] = negloglik (paras,response,predictor)

%% 'paras' is the vector of model parameters.
%% we want to estimate values for 'paras' that MINIMISE the value of the function 'negloglik'

stdev = paras(1); % first entry of 'paras' vector is the standard deviation.
% second and third entry of paras are the remaining model parameters.

mu = paras(2)+paras(3).*predictor; % here we work out the predicted mean of the response Y based on our model

probs=normpdf(response, mu, stdev); % here we work out the value of the normal p.d.f. using our predicted mean and the observed responses

out = sum(-log(probs)); % we return the negative log-likelihood 
%% This is convenient, as products turn into sums when we take logs. This means we do not have to worry about 'underflow errors'.
%% these occur when a number becomes smaller than the computer can handle, which happens frequently when multiplying many small probability densities.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% could introduce some gaps in this file that students have to fill in...



