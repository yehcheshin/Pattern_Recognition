clear all;close all; clc;

question_a();
question_b();
question_c();



function question_a()
    m1 = [1;1];
    m2 = [5;5];
    m3 = [9;1];
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
   figure(1)
    data = [data1,data2,data3,data4];
    data_label = ones(N/2,1);
    data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3]
    gscatter(data(1,:),data(2,:),data_label)
    axis equal
    
    K=3;
    for q = 2:3
    [theta,U,obj_fun] = fuzzy_kmean(data,K,q);
    [~,idx] = max(U);
    figure(q)
    
    gscatter(data(1,:),data(2,:),idx);
    title("Fuzzy k-means : q="+q);
    axis equal
    end
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
    figure(1)
    data = [data1,data2,data3,data4];
    data_label = ones(N/2,1);
    data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3]
    gscatter(data(1,:),data(2,:),data_label)
    axis equal
    K=3;
    for q = 2:3
    [theta,U,obj_fun] = fuzzy_kmean(data,K,q);
    [~,idx] = max(U);
    figure(q)
    
    gscatter(data(1,:),data(2,:),idx);
    title("Fuzzy k-means : q="+q);
   
    axis equal
    end
end
function question_c()
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
    figure(1)
    data = [data1,data2,data3,data4];
    data_label = ones(N/2,1);
    data_label = [data_label;ones(N/4,1)*2;ones(N/4,1)*3]
    gscatter(data(1,:),data(2,:),data_label)
    
    axis equal
    K = 3;
    for q = 2:3
    [theta,U,obj_fun] = fuzzy_kmean(data,K,q);
    [~,idx] = max(U);
    figure(q)
    gscatter(data(1,:),data(2,:),idx);
    hold on;
    plot(theta(:,1),theta(:,2),'xk','MarkerSize',12)
    title("Fuzzy k-means : q="+q);
    axis equal
    end
end





function [theta,U,obj_fun]=fuzzy_kmean(X,K,q)
        
    options(1) = q;
    options(2) = 100;
    options(3) = 1e-5;
    options(4) = 1;
    [theta,U,obj_fun] = fcm(X',K,options);
    
end
