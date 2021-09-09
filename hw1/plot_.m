function [] = plot_(X,pred)
    N =1000;
    for i=1:N
    if pred(i)==1
    %bayes
    %plot(X(i,1),X(i, 2),'bs');
    %maha
    %plot(X(i,1),X(i, 2),'bd');
    %Eucli
    %plot(X(i,1),X(i, 2),'bp');
    %knn
    plot(X(i,1),X(i, 2),'bp');
    elseif pred(i)==2
    %bayes
    %plot(X(i,1),X(i,2),'rs');
    %maha
    %plot(X(i,1),X(i, 2),'rd');
    %Eucli
    %plot(X(i,1),X(i, 2),'rp');
    %knn
    plot(X(i,1),X(i, 2),'rp');
    else pred(i)==3
    %bayes
    %plot(X(i,1),X(i,2),'gs');
    %maha
    %plot(X(i,1),X(i, 2),'gd');
    %Eucli
    %plot(X(i,1),X(i, 2),'gp');
    %knn
    plot(X(i,1),X(i, 2),'gp');


    end
    hold on;
    end
end
