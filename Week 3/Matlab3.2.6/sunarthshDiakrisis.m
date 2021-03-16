function [ g ] = sunarthshDiakrisis(x,m,S,p,d )
     c1=(x-m)*S^(-1);
     c2=c1*(x-m)';
     g=-0.5*c2-(d/2)*log2(2*pi)-0.5*log2(abs(det(S)))+log2(p);

end