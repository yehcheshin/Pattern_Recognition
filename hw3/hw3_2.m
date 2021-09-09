randn('seed',0);

N=100;
m1 = [2;4];
m2 = [2.5;10];
S = [1,0;0,1];
S_b = [0.25,0;0,0.25];
%(a.)
% generate data
%compute (a) FDR ratio
data1 = mvnrnd(m1,S,N)';
data2 = mvnrnd(m2,S,N)';
figure(1);
plot(data1(1,:),data1(2,:),'r.'); 
hold on;
plot(data2(1,:),data2(2,:),'b.'); 

feature_a = FDR(data1,data2)';
%(b.)
% generate data
data3 =  mvnrnd(m1,S_b,N)';
data4 =  mvnrnd(m2,S_b,N)';
figure(2);
plot(data3(1,:),data3(2,:),'r.'); 
hold on;
plot(data4(1,:),data4(2,:),'b.'); 
%compute (b) FDR ratio
feature_b = FDR(data3,data4)';



function [feature] = FDR(Data1,Data2)
    %compute the FDR ratio
    m = mean(Data1,2)-mean(Data2,2);
    variance_val = var(Data1')+var(Data2');
    for i=1:2
        feature(i) = m(i)*m(i)/variance_val(i);
    end
    
    

end

