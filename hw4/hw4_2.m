
%(a)
%set mean vector and covariance matrix  to generate data.
question_a();
question_b();
question_c();



function question_a()
m1_a = [1;1];
m2_a = [5;5];
m3_a = [9;1];
S1 = [1,0.4;0.4,1];
S2 = [1,-0.6;-0.6,1];
S3 = [1,0;0,1];
N = 500;
data1 = [];
data2 = [];
data3 = [];
data4 = [];

%generate data
 for i=1:(N/4)
    data1 = [data1,mvnrnd(m2_a,S2)'];
    data2 = [data2,mvnrnd(m2_a,S2)'];
    data3 = [data3,mvnrnd(m1_a,S1)'];
    data4 = [data4,mvnrnd(m3_a,S3)'];
 end
data = [data1,data2,data3,data4];



%random number to be a mean vector 

m_a = rand(2,1);
m_b = rand(2,1);
m_c = rand(2,1);
m_d = rand(2,1);

init_random_m = [m_a,m_b,m_c,m_d];



%plot data
figure(1);
data_label = ones(N/2,1);
data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3;];
gscatter(data(1,:),data(2,:),data_label);
case1(data,init_random_m );

end



function question_b()
m1 = [1;1];
m2 = [3.5;3.5];
m3 = [6;1];
S1 = [1,0.4;0.4,1];
S2 = [1,-0.6;-0.6,1];
S3 = [1,0;0,1];
N = 500;
data1 = [];
data2 = [];
data3 = [];
data4 = [];

%generate data
 for i=1:(N/4)
    data1 = [data1,mvnrnd(m2,S2)'];
    data2 = [data2,mvnrnd(m2,S2)'];
    data3 = [data3,mvnrnd(m1,S1)'];
    data4 = [data4,mvnrnd(m3,S3)'];
 end
data = [data1,data2,data3,data4];



m_a = rand(2,1);
m_b = rand(2,1);
m_c = rand(2,1);
m_d = rand(2,1);

init_random_m = [m_a,m_b,m_c,m_d];

%random number to be a mean vector 

%plot data
figure(1);
data_label = ones(N/2,1);
data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3;];
gscatter(data(1,:),data(2,:),data_label);
case1(data,init_random_m);

end
function  question_c()
m1 = [1;1];
m2 = [2;2];
m3 = [3;1];
S1 = [1,0.4;0.4,1];
S2 = [1,-0.6;-0.6,1];
S3 = [1,0;0,1];
N = 500;
data1 = [];
data2 = [];
data3 = [];
data4 = [];

%generate data
 for i=1:(N/4)
    data1 = [data1,mvnrnd(m2,S2)'];
    data2 = [data2,mvnrnd(m2,S2)'];
    data3 = [data3,mvnrnd(m1,S1)'];
    data4 = [data4,mvnrnd(m3,S3)'];
 end
data = [data1,data2,data3,data4];

%random choose 1 data for each sample to be a initial cluster's mean vector



m_a = rand(2,1);
m_b = rand(2,1);
m_c = rand(2,1);
m_d = rand(2,1);

init_random_m = [m_a,m_b,m_c,m_d];

%random number to be a mean vector 

%plot data
figure(1);
data_label = ones(N/2,1);
data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3;];
gscatter(data(1,:),data(2,:),data_label);
case1(data,init_random_m);

end




function case1(data,init_mean)
    for k =2:4
    if k==2
        [k2_mean,k2_idx]=Kmean(data,init_mean(:,1:k),k);
        figure(2)
     
       
        gscatter(data(1,:),data(2,:),k2_idx);
    end
    if k==3
        [k3_mean,k3_idx]=Kmean(data,init_mean(:,1:k),k);
        figure(3)
        gscatter(data(1,:),data(2,:),k3_idx);
    end
    if k==4
        [k4_mean,k4_idx]=Kmean(data,init_mean(:,1:k),k);
        figure(4)
        gscatter(data(1,:),data(2,:),k4_idx);
    end
    
end
end




%kmean algorithm 
function [m,idx]=Kmean(X,m,K)
    N = size(X,2);
    iter = 0;
    e=1;
    
    
    while(e~=0)
        iter = iter+1;
        m_old = m;
        dist = [];
        idx =[];
        for i=1:N
            each_dist = [];
            for j=1:K
                %each 
                each_dist = [each_dist;(X(:,i)-m(:,j))'*(X(:,i)-m(:,j))]; 
            end
            [val,index] =min(each_dist);
            idx=[idx,index];
        end
        
        for j=1:K
            if(sum(idx==j)~=0)
               index =find(idx==j) ;
               x_sum=[0;0];
               for i=1:size(index)
                 x_sum=x_sum+X(:,index(i))  ;
               end
               m(:,j)=x_sum;
               
            end
        end
        e = sum(sum(abs(m - m_old )));
     end
    disp(iter);
end
