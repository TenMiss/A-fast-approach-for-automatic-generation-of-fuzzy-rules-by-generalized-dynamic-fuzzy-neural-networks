function [w,y,a]=mweight(in,c,wb,out)
% This program is used to adjust the weights of TSK FNNs based 
% on simplified Mahalanobis
% Revised 11-5-2006
% Copyright Wu Shiqian.
[r,n]=size(in);
[r,u]=size(c);
[r,u]=size(wb);
[s2,n]=size(out);
a=exp(-mdist(in,c,wb));
a1=transf(a,in);
w=out/a1;
y=w*a1;