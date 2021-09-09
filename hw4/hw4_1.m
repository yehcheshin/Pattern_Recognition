



%initial suppose each cluster' prior is eaqual
P = [1/3,1/3,1/3];
N = 500;

% %question (a.)
pos_a = Question_a(P,N);
confuse_a = confuse_matrix(pos_a);

% %question (b.)

pos_b =Question_b(P,N);
confuse_b = confuse_matrix(pos_b);

% %question (c.)

pos_c = Question_c(P,N);
confuse_c = confuse_matrix(pos_c);






%EM algorithm 
function [m,S,P,pos] = EM_Algorithm(X,m,S,P,N)
    pos =[];
    for i=1:N
        pos=[pos,E_step(X(:,i),m,S,P)];
    end
    [m,S,P] = M_step(pos,X,m,S,P,N);
   

end

% Exponent step function 
function  [pos] = E_step(x,m,S,P)
    pos = [];
    for j=1:3
        pos=[pos;det(S(:,2*j-1:2*j))^(-1/2)*exp(-(x-m(:,j))'*inv(S(:,2*j-1:2*j))*(x-m(:,j))/2)*P(j)];
    end
    total_pos=sum(pos);
    pos= pos/total_pos;
end

%Maximum  step function 
function [new_mean,new_S,P] = M_step(pos,X,m,S,P,N)
    sum_pos = sum(pos,2);
    new_mean = zeros(size(m));
    new_S = zeros(size(S));
    for j=1:3
        temp_m=zeros(size(m(:,j)));
        temp_S=zeros(size(S(:,2*j-1:2*j)));
        for i=1:N
         temp_m =temp_m + pos(j,i)*X(:,i);
         temp_S =temp_S + pos(j,i)*(X(:,i)-m(:,j))*(X(:,i)-m(:,j))';
        end
        new_mean(:,j) = temp_m/sum_pos(j); 
        new_S(:,2*j-1:2*j) = temp_S/sum_pos(j);
        P(:,j) = sum_pos(j)/N;
        
        
        
       
    end
end
%generate confuse matrix
function [confuse]=confuse_matrix (pos)
   [~,idx] = max(pos);
   distribute_1 = [];
   distribute_2 = [];
   distribute_3 = [];
   
   for i=1:3
       distribute_1 = [distribute_1,sum(idx(:,1:250)==i)];
       distribute_2 = [distribute_2,sum(idx(:,251:375)==i)];
       distribute_3 = [distribute_3,sum(idx(:,376:500)==i)];
   end
   confuse = [distribute_1;distribute_2;distribute_3];
    
end


function  [pos]=Question_a(P,N)
    data1 = [];
    data2 = [];
    data3 = [];
    data4 = [];
    m1 = [1;1];
    m2 = [5;5];
    m3 = [9;1];
    S1 = [1,0.4;0.4,1];
    S2 = [1,-0.6;-0.6,1];
    S3 = [1,0;0,1];

    %generate data
     for i=1:N
         if mod(i,4)==1
             data1 = [data1,mvnrnd(m2,S2)'];
         end
         if mod(i,4)==2
             data2 = [data2,mvnrnd(m2,S2)'];
         end
         if mod(i,4)==3
             data3 = [data3,mvnrnd(m1,S1)'];
         end
         if mod(i,4)==0
             data4 = [data4,mvnrnd(m3,S3)'];
         end
         
     end
    data = [data1,data2,data3,data4];

    %plot data
    figure(1);
    data_label = ones(N/2,1);
    data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3];
    gscatter(data(1,:),data(2,:),data_label);
    ori_S = [S1,S2,S3];
    ori_m = [m1,m2,m3];
    disp("question(a.)")
    disp("original:")
    for i=1:3
        
        disp("Var"+i)
        disp(ori_S(:,2*i-1:2*i))
        disp("mean"+i)
        disp(ori_m(:,i))
    end
    
    
    axis equal;
    %generate rand mean vector and covarance matrix
    a = 1;
    b = 9;
    m_a = (b-a).*rand(2,1) + a;
    m_b = (b-a).*rand(2,1) + a;
    m_c = (b-a).*rand(2,1) + a;
    cov_a = randn(2);
    cov_b = randn(2);
    cov_c = randn(2);
    m = [m_a,m_b,m_c];
    S = [cov_a'*cov_a,cov_b'*cov_b,cov_c'*cov_c];
    %stop criterion
    error = 1e-4;
    e = sum(sum(abs(m)));
    iter = 0;
    while  e>error && iter<1000
        m_old = m;
        S_old = S;
        P_old = P;
        iter = iter+1;
        [m,S,P,pos] = EM_Algorithm(data,m,S,P,N);
        
         e = sum(sum(abs(m - m_old ))) + sum(sum(abs(S-S_old))) +sum(abs(P-P_old));
    end
    
    [~,eva_idx] = max(pos);
    figure(2)
    gscatter(data(1,:),data(2,:),eva_idx);
    axis equal;
    disp("After training")
    disp("iter:"+iter)
    disp("P(i):")
    disp(P)
    for i=1:3
        
        disp("Var"+i)
        disp(S(:,2*i-1:2*i))
        disp("mean"+i)
        disp(m(:,i))
    end
end
%question b result
function  [pos]=Question_b(P,N)
    data1 = [];
    data2 = [];
    data3 = [];
    data4 = [];
    m1 = [1;1];
    m2 = [3.5;3.5];
    m3 = [6;1];
    S1 = [1,0.4;0.4,1];
    S2 = [1,-0.6;-0.6,1];
    S3 = [1,0;0,1];

    %generate data
     for i=1:N
         if mod(i,4)==1
             data1 = [data1,mvnrnd(m2,S2)'];
         end
         if mod(i,4)==2
             data2 = [data2,mvnrnd(m2,S2)'];
         end
         if mod(i,4)==3
             data3 = [data3,mvnrnd(m1,S1)'];
         end
         if mod(i,4)==0
             data4 = [data4,mvnrnd(m3,S3)'];
         end
         
     end
    data = [data1,data2,data3,data4];
    ori_S = [S1,S2,S3];
    ori_m = [m1,m2,m3];
    disp("question(b.)")
    disp("original:")
    for i=1:3
        
        disp("Var"+i)
        disp(ori_S(:,2*i-1:2*i))
        disp("mean"+i)
        disp(ori_m(:,i))
    end
    %plot data
    figure(3);
    data_label = ones(N/2,1);
    data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3];
    gscatter(data(1,:),data(2,:),data_label);
    axis equal;
    
    
    a = 1;
    b = 6;
    m_a = (b-a).*rand(2,1) + a;
    m_b = (b-a).*rand(2,1) + a;
    m_c = (b-a).*rand(2,1) + a;
    
    cov_a = randn(2);
    cov_b = randn(2);
    cov_c = randn(2);
   
    
    m = [m_a,m_b,m_c];
    S = [cov_a'*cov_a,cov_b'*cov_b,cov_c'*cov_c];
    error = 1e-5;
    e = sum(sum(abs(m)));
    iter = 0;
    while  e>error && iter<1000
        m_old = m;
        S_old = S;
        P_old = P;
        iter = iter+1;
        [m,S,P,pos] = EM_Algorithm(data,m,S,P,N);
         e = sum(sum(abs(m - m_old ))) + sum(sum(abs(S-S_old))) +sum(abs(P-P_old));
    end
    
    [~,eva_idx] = max(pos);
    figure(4)
    gscatter(data(1,:),data(2,:),eva_idx);
    axis equal;
    disp("After training")
    disp("iter:"+iter)
    disp("P(i):")
    disp(P)
    for i=1:3
        
        disp("Var"+i)
        disp(S(:,2*i-1:2*i))
        disp("mean"+i)
        disp(m(:,i))
    end
end
%question c result 
function [pos] = Question_c(P,N)
    data1 = [];
    data2 = [];
    data3 = [];
    data4 = [];
    m1 = [1;1];
    m2 = [2;2];
    m3 = [3;1];
    S1 = [1,0.4;0.4,1];
    S2 = [1,-0.6;-0.6,1];
    S3 = [1,0;0,1];

    %generate data
     for i=1:N
         if mod(i,4)==1
             data1 = [data1,mvnrnd(m2,S2)'];
         end
         if mod(i,4)==2
             data2 = [data2,mvnrnd(m2,S2)'];
         end
         if mod(i,4)==3
             data3 = [data3,mvnrnd(m1,S1)'];
         end
         if mod(i,4)==0
             data4 = [data4,mvnrnd(m3,S3)'];
         end
         
     end
    data = [data1,data2,data3,data4];
    ori_S = [S1,S2,S3];
    ori_m = [m1,m2,m3];
    disp("question(a.)")
    disp("original:")
    for i=1:3
        
        disp("Var"+i)
        disp(ori_S(:,2*i-1:2*i))
        disp("mean"+i)
        disp(ori_m(:,i))
    end
    %plot data
    figure(5);
    data_label = ones(N/2,1);
    data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3];
    gscatter(data(1,:),data(2,:),data_label);
    axis equal;
    cov_a = randn(2);
    cov_b = randn(2);
    cov_c = randn(2);
    a = 1;
    b = 3;
    m_a = (b-a).*rand(2,1) + a;
    m_b = (b-a).*rand(2,1) + a;
    m_c = (b-a).*rand(2,1) + a;
    m = [m_a,m_b,m_c];
    S = [cov_a'*cov_a,cov_b'*cov_b,cov_c'*cov_c];
    error = 1e-5;
    e = sum(sum(abs(m)));
    iter = 0;
    
    
    
    while  e>error && iter<1000
        m_old = m;
        S_old = S;
        P_old = P;
        iter = iter+1;
        if (det(S(:,1:2))*det(S(:,3:4))*det(S(:,5:6)))==0
            break
        end
        [m,S,P,pos] = EM_Algorithm(data,m,S,P,N);
       
        e = sum(sum(abs(m - m_old ))) + sum(sum(abs(S-S_old))) +sum(abs(P-P_old));
    end
    
    [~,eva_idx] = max(pos);
    figure(6)
    gscatter(data(1,:),data(2,:),eva_idx);
    axis equal;
    disp("After training")
    disp("iter:"+iter)
    disp("P(i):")
    disp(P)
    for i=1:3
        
        disp("Var"+i)
        disp(S(:,2*i-1:2*i))
        disp("mean"+i)
        disp(m(:,i))
    end

end