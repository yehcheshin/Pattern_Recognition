
N=1000;
w = [1;1];
w0 =0;
a=10;
e=1;
%(a.)generate data 
X = generate_hyper(w,w0,10,1,N,0);
m = mean(X,2);
%compute covarivance matrix
X = X - m;
cov_mat = X*X';
%computer eigenvale and eigen vector
[W,D] = eig(cov_mat);
Z = W'*X;

eigenvalue = diag(D);
%sorted eigenvalue 
[eigenvalue,idx] = sort(eigenvalue,1,'descend');
eigenvector = W(:,idx);
explain = eigenvalue / sum(eigenvalue);
%select first principle component 
eigenvalue = eigenvalue(1:2);
eigenvector = eigenvector(:,1:2);

A = eigenvector';
X_PCA = A*X;

hold on;
plot(X_PCA(1,:),X_PCA(2,:),'r.'); 





function X = generate_hyper(w,w0,a,e,N,sed)
    randn('seed',sed)
    l = size(w,1);
    t= (rand(l-1,N)-.5)*2*a;
    t_last = -(w(1:l-1)/w(l))'*t + 2*e*(rand(1,N)-.5)-(w0/w(1));
    X = [t;t_last];
    
    if(l==2)
        figure(1),plot(X(1,:),X(2,:),'b.');
    elseif(l==3)
        figure(1),plot3(X(1,:),X(2,:),X(3,:),'.b');
    end
    figure(1),axis equal

end