% (a.)小題 data生成
randn('seed',0)
P  =[1/3 1/3 1/3];% Prior(C1)== Prior(C2)== Prior(C3)==1/3
P2 = [0.8 0.1 0.1]; % Prior(C1)=0.8 Prior(C2)=0.1 Prior(C3)=0.1
N = 1000;%N=1000筆data (C1:333 C2:333 C3:334)
m1=[1;1];%mean vector:m1
S1=[2 0;0 2];%covariance matrices S1
m2=[4;4];%mean vector:m2
S2=[2 0;0 2];%covariance matrices S2
m3=[8;1];%mean vector:m3
S3=[2 0;0 2];%covariance matrices S3

%根據 mi Si 產生 (N * Prior(i))個 random data 
data1=mvnrnd(m1, S1, fix(N*P(1))); 
data2=mvnrnd(m2,S2,fix(N*P(2)));
data3=mvnrnd(m3,S3,fix(N*P(3))+1);
data2_1 = mvnrnd(m1, S1, fix(N*P2(1))); 
data2_2=mvnrnd(m2,S2,fix(N*P2(2)));
data2_3=mvnrnd(m3,S3,fix(N*P2(3))+1);


%產生ground truth label 
label_c1 =[ones(1,fix(N*P(1)))]; 
label_c2 =[ones(1,fix(N*P(2)))+1];
label_c3 =[ones(1,fix(N*P(3)+1))+2]; 
label2_c1= [ones(1,fix(N*P2(1)))]; 
label2_c2 =[ones(1,fix(N*P2(2)))+1];
label2_c3 =[ones(1,fix(N*P2(3)+1))+2]; 

%X5 :X5 dataset Nx2 :N筆1x2維的data座標
%X5_ :X5' dataset Nx2 :N筆1x2維的data座標
X5 = [data1;data2;data3];
x5_ = [data2_1;data2_2;data2_3];

%y_label: X5's label  1xN :N筆data之grount truth label
%y_label2: X5_'s label 1xN :N筆data之grount truth label
y_label = [label_c1 label_c2 label_c3];
y_label2 = [label2_c1 label2_c2 label2_c3];

%X5 3個classes bayesian classifier's discrimintation function 結果
bayesian_g1 = Bayesian_classifier(m1,S1,P(1),X5);
bayesian_g2 = Bayesian_classifier(m2,S2,P(2),X5);
bayesian_g3 = Bayesian_classifier(m3,S3,P(3),X5);
bayesian_g = [bayesian_g1;bayesian_g2;bayesian_g3];
%X5_ %3個classes bayesian classifier's discrimintation function 結果
bayesian2_g1 = Bayesian_classifier(m1,S1,P2(1),x5_);
bayesian2_g2 = Bayesian_classifier(m2,S2,P2(2),x5_);
bayesian2_g3 = Bayesian_classifier(m3,S3,P2(3),x5_);
bayesian2_g = [bayesian2_g1;bayesian2_g2;bayesian2_g3];


%3個classes Euclidean classifier's discrimintation function 結果
Euclidean_g1 =  Euclidean_classifier(m1,S1,P(1),X5);
Euclidean_g2 =  Euclidean_classifier(m2,S2,P(2),X5);
Euclidean_g3 =  Euclidean_classifier(m3,S3,P(3),X5);
Euclidean_g = [Euclidean_g1;Euclidean_g2;Euclidean_g3];
%3個classes Euclidean classifier's discrimintation function 結果
Euclidean2_g1 =  Euclidean_classifier(m1,S1,P2(1),x5_);
Euclidean2_g2 =  Euclidean_classifier(m2,S2,P2(2),x5_);
Euclidean2_g3 =  Euclidean_classifier(m3,S3,P2(3),x5_);
Euclidean2_g = [Euclidean2_g1;Euclidean2_g2;Euclidean2_g3];

%bayesian predicted: 每筆data 逐一比較三個 bayesian_g(i)結果，取最大值即predicted class
[M,bayesian_pred] =  max(bayesian_g);
[M,bayesian2_pred] =  max(bayesian2_g);
%Euclidean predicted: 每筆data 逐一比較三個 Euclidean(i)結果，取最小值即predicted class
[M,Euclidean_pred] = min(Euclidean_g);
[M,Euclidean2_pred] = min(Euclidean2_g);



%bayesian/Mahalanobis/Euclidean 的error compute 
error_bayes = 0;
error_Eucli =0;
error_bayes2 = 0;
error_Eucli2 =0;
for i=1:size(y_label,2)
    if y_label(i) ~= bayesian_pred(i)
        error_bayes = error_bayes +1;
    end
    
    if y_label(i) ~= Euclidean_pred(i)
        error_Eucli = error_Eucli+1;
    end
    if y_label2(i) ~= bayesian2_pred(i)
         error_bayes2 = error_bayes2 +1;
    end
    if y_label2(i) ~= Euclidean2_pred(i)
        error_Eucli2 = error_Eucli2+1;
    end
end

n =size(X5,1);
error_bayes = error_bayes/n
error_bayes2 = error_bayes2/n
error_Eucli =error_Eucli/n
error_Eucli2 = error_Eucli2/n




 plot(data1(:,1),data1(:, 2),'b.');
 hold on;
 plot(data2(:,1),data2(:,2),'r.');
 plot(data3(:,1),data3(:,2),'g.');
% plot_(X5,Euclidean_pred);
%  plot(data2_1(:,1),data2_1(:, 2),'b.');
%  hold on;
%  plot(data2_2(:,1),data2_2(:,2),'r.');
%  plot(data2_3(:,1),data2_3(:,2),'g.');
% plot_(x5_,Euclidean2_pred);


 
%Bayesian classifier 
function [g] = Bayesian_classifier(m,S,prior,x)
n = size(x,1);
iso_cov = diag(S); %對角項之元素
iso_cov = iso_cov(1,:);


for i=1:n
    g(i) = (1/iso_cov)*(m')*(x(i,:)')-(1/iso_cov/2)*(m')*m + log(prior);

%     g(i) = (1/iso_cov*m')*x(i,:)'+((-1/(2*iso_cov))* m'*m+log(prior));
end
end

%Euclidean classifier
function [g] = Euclidean_classifier(m,S,prior,x)
n = size(x,1);
for i =1:n
    g(i) = sqrt((x(i,:)'- m)'*(x(i,:)'-m));
end
end
