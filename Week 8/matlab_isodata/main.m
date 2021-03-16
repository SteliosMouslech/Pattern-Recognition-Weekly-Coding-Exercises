clear all;
clc;
fid = fopen('iris.data','r');
i=1;
CurrLine = 'start';
while(~feof(fid))
    CurrLine = fgetl(fid);
    if(~isempty(CurrLine))
        patterns(1,i)     =  str2double( CurrLine(1:3)   ) ;  
        patterns(2,i)     =  str2double( CurrLine(5:7)   ) ;  
        patterns(3,i)     =  str2double( CurrLine(9:11)  ) ;  
        patterns(4,i)     =  str2double( CurrLine(13:15) ) ;  
        tmp                =  CurrLine(17:length(CurrLine));  
        if strcmp(tmp,'Iris-setosa')
            targets(i) = 0;
        elseif strcmp(tmp,'Iris-versicolor')
             targets(i) = 1;
        elseif strcmp(tmp,'Iris-virginica')
            targets(i) = 2;
        else 
            targets(i) = -1;
        end
        i = i+1;
    end
end
fclose(fid);
close all;
clc;
ON=4; % threshold number of elements for the elimination of a cluster.
OC=4; % threshold distance for the union of clusters.
OS=0.5;  % deviation typical threshold for the division of a cluster.
k=5;   % number (maximum) cluster.
L=1;   % maximum number of clusters that can be mixed in a single iteration.
I=150;  % maximum number of iterations allowed.
NO=1;  % extra parameter to automatically answer no to the request of cambial any parameter.
min_dist=4; % Minimum distance a point must be in each center. If you want to delete any point
        % Give a high value.
%%%%%%%%%%%%%%%%%%%%%
%  Function ISODATA  %
%%%%%%%%%%%%%%%%%%%%%
[centro, Xcluster, A, clusters]=isodata_ND(patterns', k, L, I, ON, OC, OS, NO, min_dist);

fprintf('Number of Clusters: %d\n',A);
counter = zeros(3,3);
k= zeros(1,3);
for i=0:2
    for j=1:50
        counter(i+1,clusters(i*50 + j))= counter(i+1,clusters(i*50 + j)) + 1;
    end
    %[~,k(i+1)] = max(counter);
end

[maxes,id] = max(counter);
maxid = sortrows([maxes; [1 2 3]; id]',-1);

clustersfixed=zeros(150,1);
for i=1:3
    clustersfixed(clusters==i) = id(i) - 1;
end

error_rate=nnz(clustersfixed'~=targets)/150;
fprintf('Error rate is: %d\n',error_rate);

