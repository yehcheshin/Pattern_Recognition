% (a.)小題 data生成
randn('seed',0)
P = 1/3; % Prior(C1) == Prior(C2) == Prior(C3) == 1/3
N = 1000;% N=1000筆data (C1:333 C2:333 C3:334)
m1=[1;1]; %mean vector:m1
S1=[6 0;0 6];%covariance matrices S1
m2=[8;6];%mean vector:m2
S2=[6 0;0 6];%covariance matrices S2
m3=[13;1];%mean vector:m3
S3=[6 0;0 6];%covariance matrices S3
%生成X3dataset作為test set
data1=mvnrnd(m1, S1, fix(N*P)); 
data2=mvnrnd(m2,S2,fix(N*P));
data3=mvnrnd(m3,S3,fix(N*P+1));
data = [data1;data2;data3];
%生成X3dataset label
label_c1 =[ones(1,fix(N*P))]; 
label_c2 =[ones(1,fix(N*P))+1]; 
label_c3 =[ones(1,fix(N*P+1))+2]; 
y_label = [label_c1 label_c2 label_c3];

%將Z換一個random seed 來產生資料
randn('seed',100)
%生成Z dataset(使用X3相同的mean及covariance)
z1=mvnrnd(m1, S1, fix(N*P)); 
z2=mvnrnd(m2,S2,fix(N*P));
z3=mvnrnd(m3,S3,fix(N*P+1));
z_data = [z1;z2;z3];
%生成Z dataset 之label 
labelz_c1 =[ones(1,fix(N*P))]; 
labelz_c2 =[ones(1,fix(N*P))+1]; 
labelz_c3 =[ones(1,fix(N*P+1))+2]; 
z_label = [labelz_c1 labelz_c2 labelz_c3];

%k=1,k=11
k = [1 11];

%knn的error compute 
for i=1:2   
    knn_predict(i,:) = knn_classifier(z_data,z_label,k(i),data);
end

error_knn =[0 0];

for i=1:size(y_label,2)
    for j=1:2
    if y_label(i) ~= knn_predict(j,i)
        error_knn(j) = error_knn(j) +1;
    end
  
    end
end
% error_knn  1x2 [k=1之error k=11之error ]
error_knn =error_knn/N


% 將X3 dataset 視覺化
plot(data1(:,1),data1(:, 2),'b.');
hold on;
plot(data2(:,1),data2(:,2),'r.');
plot(data3(:,1),data3(:,2),'g.');
plot_(data,knn_predict(1,:));
%將Z dataset 視覺化
% plot(z1(:,1),z1(:, 2),'y^');
% hold on;
% plot(z2(:,1),z2(:,2),'k^');
% plot(z3(:,1),z3(:,2),'m^4');
 
%knn classifier
function [g] = knn_classifier(z,label,k,data)
n = size(z,1);
n1 = size(data,1);
c = max(label);
    for i=1:n1
        dist = sum((data(i,:)'*ones(1,n)-z').^2);
        [sorted,nearst] = sort(dist);% 排序 data_i與所有Z的data距離（由小排到大），取K個
        class_count =zeros(1,c);
        for q=1:k% 計算前K個點中哪個類別數最多
            class = label(nearst(q));
            class_count(class) = class_count(class)+1;
        end%% 類別數最多者即預測之類別
        [val,g(i)] = max(class_count);
    end
end


