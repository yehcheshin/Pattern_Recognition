N = 1000;
%(a)
m_a = [1;1]; %(a) mean vector:  
cov_a  = [5,3;3,4];%(a)convariance matrix 
data_A=mvnrnd(m_a,cov_a,N)';%依mean vector 與 convariance matrix 生成1000筆
figure(1);
plot(data_A(1,:),data_A(2,:),'b.');
hold on;


%(b)
m_b = [10;5];%(b) mean vector:  
cov_b  = [7,4;4,5];%(b)convariance matrix 
data_B=mvnrnd(m_b,cov_b,N)';%依mean vector 與 convariance matrix 生成1000筆
[A_m,A_S] = MLE(data_A,N)%已知:data_A 經過Maximun Likelihood Estimation求出學習後的mean與convariance
[B_m,B_S] = MLE(data_B,N)%已知:data_A 經過Maximun Likelihood Estimation求出學習後的mean與convariance
data_hat_A = mvnrnd(A_m,A_S,N);%將估算完的hat_mean,hat_S生成data 與原本S及m做比較 
data_hat_B = mvnrnd(B_m,B_S,N);%將估算完的hat_mean,hat_S生成data 與原本S及m做比較 
plot(data_hat_A(:,1),data_hat_A(:, 2),'r.');
% figure(2);
%plot(data_B(1,:),data_B(2,:),'b.');
% plot(data_hat_B(:,1),data_hat_B(:, 2),'r.');

function [hat_m,hat_S] = MLE(X,N)
    %  Multivariate Normal Distribution (m,S 為未知數)
    hat_m = sum(X,2)/N; %求出 未知數 mean's ml: hat_m 
    hat_S=zeros(2);
    for i=1:N
        hat_S =hat_S+((X(:,i)-hat_m)*(X(:,i)-hat_m)'); %將求出的mean_ML 帶入M，並對convarance matrix S偏微分，求出hat_S
    end
    hat_S = hat_S/N;
  
end