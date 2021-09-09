% (a.)小題 data生成
randn('seed',0);
%classes數：3（C1,C2,C3）
P = 1/3; % Prior(C1) == Prior(C2) == Prior(C3) == 1/3
N = 1000;% N=1000筆data (C1:333 C2:333 C3:334)
m1=[1;1];%mean vector:m1
S1=[6 0;0 6];%covariance matrices S1
m2=[8;6];%mean vector:m2
S2=[6 0;0 6];%covariance matrices S2
m3=[13;1];%mean vector:m3
S3=[6 0;0 6];%covariance matrices S3

%根據 mi Si 產生 (N * Prior(i))個 random data 

data1=mvnrnd(m1, S1, fix(N*P));
data2=mvnrnd(m2,S2,fix(N*P));
data3=mvnrnd(m3,S3,fix(N*P)+1);

%產生ground truth label 
label_c1 =[ones(1,fix(N*P))];%class1 label :1
label_c2 =[ones(1,fix(N*P))+1]; %class2 label:2 
label_c3 =[ones(1,fix(N*P)+1)+2]; %class3 label:3 

%data(X3 data): Nx2 :N筆1x2維的data座標
%y_label: 1xN :N筆data之grount truth label
X3 = [data1;data2;data3];
y_label = [label_c1 label_c2 label_c3];

% (b.)小題 apply bayesian/Mahalanobis/Euclidean clasifier on X1

%3個classes bayesian classifier's discrimintation function 結果
bayesian_g1 = Bayesian_classifier(m1,S1,P,X3);
bayesian_g2 = Bayesian_classifier(m2,S2,P,X3);
bayesian_g3 = Bayesian_classifier(m3,S3,P,X3);
bayesian_g = [bayesian_g1;bayesian_g2;bayesian_g3];

%3個classes Mahalanobis classifier's discrimintation function 結果
Mahalanobis_g1 = Mahalanobis_classifier(m1,S1,P,X3);
Mahalanobis_g2 = Mahalanobis_classifier(m2,S2,P,X3);
Mahalanobis_g3 = Mahalanobis_classifier(m3,S3,P,X3);
Mahalanobis_g = [Mahalanobis_g1;Mahalanobis_g2;Mahalanobis_g3];

%3個classes Euclidean classifier's discrimintation function 結果
Euclidean_g1 =  Euclidean_classifier(m1,S1,P,X3);
Euclidean_g2 =  Euclidean_classifier(m2,S2,P,X3);
Euclidean_g3 =  Euclidean_classifier(m3,S3,P,X3);
Euclidean_g = [Euclidean_g1;Euclidean_g2;Euclidean_g3];



%bayesian predicted: 每筆data 逐一比較三個 bayesian_g(i)結果，取最大值即predicted class
[M,bayesian_pred] =  max(bayesian_g);
%Mahalanobis predicted: 每筆data 逐一比較三個 Mahalanobis(i)結果，取最小值即predicted class
[M,Mahalanobis_pred ] = min(Mahalanobis_g);
%Euclidean predicted: 每筆data 逐一比較三個 Euclidean(i)結果，取最小值即predicted class
[M,Euclidean_pred] = min(Euclidean_g);


%bayesian/Mahalanobis/Euclidean 的error compute 
error_bayes = 0;
error_Maha =0;
error_Eucli =0;
for i=1:size(y_label,2)
    if y_label(i) ~= bayesian_pred(i)
        error_bayes = error_bayes +1;
    end
    if y_label(i) ~= Mahalanobis_pred(i)
        error_Maha = error_Maha+1;
    end
    if y_label(i) ~= Euclidean_pred(i)
        error_Eucli = error_Eucli+1;
    end
end
n =size(X3,1);
error_bayes = error_bayes/n
error_Maha =error_Maha/n
error_Eucli =error_Eucli/n





 plot(data1(:,1),data1(:, 2),'b.');
 hold on;
 plot(data2(:,1),data2(:,2),'r.');
 plot(data3(:,1),data3(:,2),'g.');
%  plot_(X3,Mahalanobis_pred);


%Bayesian classifier 
function [g] = Bayesian_classifier(m,S,prior,x)
n = size(x,1);
iso_cov = diag(S); %對角項之元素
iso_cov = iso_cov(1,:);
for i=1:n
    %isotropic covariance
    g(i) = (1/iso_cov*m')*x(i,:)'+((-1/(2*iso_cov))* m'*m+log(prior));
end
end

%Euclidean classifier
function [g] = Euclidean_classifier(m,S,prior,x)
n = size(x,1);
for i =1:n
    g(i) = sqrt((x(i,:)'- m)'*(x(i,:)'-m));
end
end

%Mahalanobis classifier
function [g] = Mahalanobis_classifier(m,S,prior,x)
n = size(x,1);
for i=1:n
    g(i) = sqrt((x(i,:)'-m)'*inv(S)*(x(i,:)'-m));
end
end
