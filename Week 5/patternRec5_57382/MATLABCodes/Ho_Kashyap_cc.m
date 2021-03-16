function [a, b] = Ho_Kashyap_cc(train_features, train_targets, type, Max_iter, b_min, eta) 
 
% Classify using the using the Ho-Kashyap algorithm 
% Inputs: 
% 	features	- Train features 
%	targets	    - Train targets 
%	Type(0 Basic/1 Modified), Maximum iteration, Convergence criterion, Convergence rate 
% 
% Outputs 
%   a           - Classifier weights 
%   b           - Margin 
 
[c, n]		   = size(train_features); 
train_class2   = find(train_targets == -1); 
 
%Preprocessing (Needed so that b>0 for all features) 
Y = train_features; 
Y(:,train_class2) = -Y(:,train_class2); 
 
b                  = ones(1,n); 
a                  = pinv(Y')*b'; 
k	               = 0; 
e    	           = 1e3; 
found              = 0; 
 
while ((sum(abs(e) > b_min)>0) & (k < Max_iter) &(~found)) 
    k = k+1; 
    e  = (Y' * a)' - b; 
    e_plus  = 0.5*(e + abs(e)); 
    b = b + 2*eta*e_plus; 
     
    if (type==0), 
        a = pinv(Y')*b'; 
    else 
        a = a + eta*pinv(Y')*e_plus'; 
    end         
end 
 
if (k == Max_iter), 
   disp(['No solution found']); 
else 
   disp(['Did ' num2str(k) ' iterations']) 
end 
 
