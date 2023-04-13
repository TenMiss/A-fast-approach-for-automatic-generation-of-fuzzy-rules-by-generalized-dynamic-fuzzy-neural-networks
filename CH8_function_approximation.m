% Nonlinear Dynamic System Identification in chapter 8 by GD-FNN
% Revised 11-5-2006
% Copyright Wu Shiqian.clear all
close all
clear all
rand('seed',50);
p=rand(3,216)*5+1;
t=(1+p(1,:).^0.5+p(2,:).^(-1)+p(3,:).^(-1.5)).^2;
[r,q]=size(p);
[s2,q]=size(t);
pp=p';
p1=min(pp);
p2=max(pp);
range=[p1' p2'];
scope=(range(:,2)-range(:,1));
% Setting initial values
width0=scope*ones(1,2)/2;
kdmax=sqrt(log(1/0.5));
kdmin=sqrt(log(1/0.8));
gama=(kdmin/kdmax)^(3/q);
emax=max(t)/2;
emin=0.8;
beta=(emin/emax)^(3/q);
width0=scope*ones(1,2)/2;
k=4;
km=0.65;
ks=0.9;
kerr=0.002;
parameters(1)=kdmax;
parameters(2)=kdmin;
parameters(3)=gama;
parameters(4)=emax;
parameters(5)=emin;
parameters(6)=beta;
parameters(7)=k;
parameters(8)=km;
parameters(9)=ks;
parameters(10)=kerr;
[CRBF, width, rule, e, RMSE] = GDFNN(p, t, width0, parameters);
[r,u]=size(CRBF);
[w2,A1,A]=mweight(p,CRBF,width,t);
derr=abs(t-A1);
APEtrain=sum(derr./abs(t))/q

figure
plot(rule,'r');
title('Fuzzy rule generation');
xlabel('sample patterns');

figure
plot(e,'r');
title('Actual output error');
xlabel('sample patterns');

figure
plot(RMSE,'r');
title('Root mean squared error');
xlabel('sample patterns');

figure;   
plot(p,t,'r.',p,A1,'g*');
title('comparison between desired and actual outputs');
xlabel('input data p');
ylabel('desired & actual output')