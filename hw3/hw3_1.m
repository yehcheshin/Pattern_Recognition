%（a_i) generate data with mean ,covariance matrix
N = 100;
m1_a = [-10;-10]; 
m2_a = [-10;10];
m3_a = [10;-10];
m4_a = [10;10];
m_a = [m1_a,m2_a,m3_a,m4_a];
p = 1/4;
c = 4; 
S = [0.2,0;0,0.2];
% generate data
data_a1 = mvnrnd(m1_a,S,N)';%class 1
data_a2 = mvnrnd(m2_a,S,N)';%class 2
data_a3 = mvnrnd(m3_a,S,N)';%class 3
data_a4 = mvnrnd(m4_a,S,N)';%class 4
figure(1);
plot(data_a1(1,:),data_a1(2,:),'b.');
hold on;
plot(data_a2(1,:),data_a2(2,:),'g.');
plot(data_a3(1,:),data_a3(2,:),'r.');
plot(data_a4(1,:),data_a4(2,:),'k.');

data_A = [data_a1,data_a2,data_a3,data_a4];

%(a_ii) compute the Scatter_w,Scatter_b,Scatter_m matrices

Scatter_w_a = Gen_Sw(data_A,m_a,S,N,c);%compute S_w
Scatter_b_a = Gen_Sb(data_A,m_a,N,p,c);%compute S_b
Scatter_m_a = Scatter_w_a + Scatter_b_a;%compute S_a
%(a_iii) compute the J3
J3_a =trace(inv(Scatter_w_a)*Scatter_m_a);


%(b_i) generate data with mean ,covariance matrix
m1_b = [-1;-1]; 
m2_b = [-1;1];
m3_b = [1;-1];
m4_b = [1;1];
m_b = [m1_b,m2_b,m3_b,m4_b];

% generate data
data_b1 = mvnrnd(m1_b,S,N)';%class 1
data_b2 = mvnrnd(m2_b,S,N)';%class 2
data_b3 = mvnrnd(m3_b,S,N)';%class 3
data_b4 = mvnrnd(m4_b,S,N)';%class 4
figure(2);
hold on;
plot(data_b1(1,:),data_b1(2,:),'b.');
plot(data_b2(1,:),data_b2(2,:),'g.');
plot(data_b3(1,:),data_b3(2,:),'r.');
plot(data_b4(1,:),data_b4(2,:),'k.');



data_B = [data_b1,data_b2,data_b3,data_b4];
%(b_ii) compute the Scatter_w,Scatter_b,Scatter_m matrices
Scatter_w_b = Gen_Sw(data_B,m_b,S,N,c);%compute S_w
Scatter_b_b = Gen_Sb(data_B,m_b,N,p,c);%compute S_b
Scatter_m_b = Scatter_w_b + Scatter_b_b;%compute S_a
%(b_iii) compute the J3
J3_b =trace(inv(Scatter_w_b)*Scatter_m_b);

%(c_i)generate data with mean ,covariance matrix
S_c = [3,0;0,3];
% generate data
data_c1 = mvnrnd(m1_a,S_c,N)';%class 1
data_c2 = mvnrnd(m2_a,S_c,N)';%class 2
data_c3 = mvnrnd(m3_a,S_c,N)';%class 3
data_c4 = mvnrnd(m4_a,S_c,N)';%class 4

figure(3);
hold on;
plot(data_c1(1,:),data_c1(2,:),'b.');
plot(data_c2(1,:),data_c2(2,:),'g.');
plot(data_c3(1,:),data_c3(2,:),'r.');
plot(data_c4(1,:),data_c4(2,:),'k.');

data_C = [data_c1,data_c2,data_c3,data_c4];

%(c_ii) compute the Scatter_w,Scatter_b,Scatter_m matrices

Scatter_w_c = Gen_Sw(data_C,m_a,S_c,N,c);%compute S_w
Scatter_b_c = Gen_Sb(data_C,m_a,N,p,c);%compute S_b
Scatter_m_c = Scatter_w_c + Scatter_b_c;%compute S_a
%(a_iii) compute the J3
J3_c =trace(inv(Scatter_w_c)*Scatter_m_c);




function [S_w] = Gen_Sw(X,m,S,N,class)
   S_w =[0,0;0,0];
   
    for c=1:class
        temp = X(:,(c-1)*100+1:c*100) - m(:,c);
        for i=1:N
            S_w =S_w +temp(:,i)*temp(:,i)';
        end
    end
    S_w = S_w/(N*c);




end

function [S_b] = Gen_Sb(X,m,N,p,class)
   S_b =[0,0;0,0];
   total_mean = mean(X,2);
    for c=1:class
         S_b = S_b + p*(m(:,c) - total_mean)*(m(:,c) - total_mean)';
        
    end
    
end
