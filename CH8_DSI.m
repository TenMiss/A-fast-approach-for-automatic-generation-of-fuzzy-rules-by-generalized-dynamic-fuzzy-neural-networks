% Nonlinear Dynamic System Identification in chapter 8 by GD-FNN
% Revised 11-5-2006
% Copyright Wu Shiqian.
clear all
clear all
y=zeros(1,200);
y(1)=0;
y(2)=1;
a=sin(2*pi/25);
p(:,1)=[1;0;a];
t(1)=sin(2*pi*2/25);
for k=3:200
    y(k+1)=(y(k)*y(k-1)*(y(k)+2.5))./(1+y(k).^2+y(k-1).^2)+sin(2*pi*k/25);
    x1(k-1)=y(k);
    x2(k-1)=y(k-1);
    x3(k-1)=sin(2*pi*k./25);
    p=[x1;x2;x3];
    t(k-1)=y(k+1);
end
[r,q]=size(p);
[s2,q]=size(t);
pp=p';
p1=min(pp);
p2=max(pp);
range=[p1' p2'];
scope=(range(:,2)-range(:,1));
% Setting initial values
kdmax=sqrt(log(1/0.5));
kdmin=sqrt(log(1/0.8));
gama=(kdmin/kdmax)^(3/q);
emax=1;
emin=0.03;
beta=(emin/emax)^(3/q);
width0=scope*ones(1,2)/2;
k=1.35;
km=0.8;
ks=0.8;
kerr=0.01;
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
kk=1:199;
plot(kk,t,'r-',kk,A1,'ro');
title('Comparison between desired and actual outputs');
xlabel('Time t');