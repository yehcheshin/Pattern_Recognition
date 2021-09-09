N = 1000;
%(a),(b) 之 prior
p = [0.25,0.5];
for i=1:2
    P(i,:) = [ones(1,N)*p(i)];
end
n = [ones(1,N)];
%產生(a)之sample
A = binornd(n,P(1));

%產生(b)之sample
B = binornd(n,P(2));
%運算Bernoulli MLE
p_ml_A = Bernoulli_MLE(A,N)
p_ml_B = Bernoulli_MLE(B,N)

%Bernouli estimator
function [hat_p] = Bernoulli_MLE(X,N)
     syms p%未知prior
     L = log(p)*sum(X)+log(1-p)*(N-sum(X));%Bernoulli log likelihood estimator
     hat_p =double(solve(diff(L,p,1)==0));%prior_ML 
     
end