%% Ho Kashyap
[X, ~] = iris_dataset;
X = X(:,51:150);
y = [ ones(1,50) -(ones(1,50))];
X1=[X;ones(1,100)];
[w,b]=Ho_Kashyap_cc(X1,y, 0, 1000, 0.1, 0.01);

disp('weights are:')
w
disp('error rate is')
KH_out=2*(w'*X1>0)-1;
err_KH=sum(KH_out.*y<0)/100