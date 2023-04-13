function [CRBF, width, rule, e, RMSE] = GDFNN(p, t, width0, parameters)
% This is GD-FNN training program.
% Input:
%       p is the input data, which is r by q matrix. r is the No. of input
%       t is the output data,which is s2 by q matrix. q is the No. of sample data.
%       parameters is a vector which defines the predefined parameters 
%       parameters(1)= kdmax     
%       parameters(2)= kdmin     
%       parameters(3)= gama      
%       parameters(4)= emax      
%       parameters(5)= emin     
%       parameters(6)= beta      
%       parameters(7)= k        
%       parameters(8)= km        
%       parameters(9)= ks      
%       parameters(10)= kerr     
% Output:
%       CRBF is the centers of the RBF units, which is a r by u matrix
%       width is the widths of RBF units, which is a r by u matrix
%       rule is the number of rules for each iteration
%       e is the output error for each iteration
%       RMSE is the root mean squared error for each iteration
% Revised 11-5-2006
% Copyright Wu Shiqian.
if nargin<4
    error('Not enough input arguments')
end
if size(p,2)~=size(t,2)
    error('The input data are not correct')
end
[r,q]=size(p);
[s2,q]=size(t);
pp=p';
p1=min(pp);
p2=max(pp);
range=[p1' p2'];
scope=(range(:,2)-range(:,1));
% Setting predefined parameters
kdmax=parameters(1);
kdmin=parameters(2);
gama=parameters(3);
emax=parameters(4);
emin=parameters(5);
beta=parameters(6);
k=parameters(7);
km=parameters(8);
ks=parameters(9);
kerr=parameters(10);
ALLIN=[];
ALLOUT=[];
CRBF=[];
width=[];
%When first sample data coming
ALLIN=p(:,1);
ALLOUT=t(:,1);
% Seting up the initial FNN
diffc=abs(range-p(:,1)*ones(1,2));
for j=1:r
   cvar=abs(diffc(j,:));
   [cdmin,nmin]=min(cvar);
   if cdmin<=km
      CRBF(j,1)=range(j,nmin);
      width(j,1)=width0(j,nmin);
   else
      [cdb,nb]=max(cvar);
      CRBF(j,1)=p(j,1);
      width(j,1)=cdb/0.8;
   end
end
w1=CRBF';
rule(1)=1;
% Caculating the first out error
a0=exp(-mdist(ALLIN,CRBF,width));
a01=[a0 p(:,1)'];
w2=ALLOUT/a01';
a02=w2*a01';
sse(1)=sumsqr(ALLOUT-a02)/s2;
rmse(1)=sqrt(sse(1));
% When ith sample data coming
for i=2:q
   IN=p(:,i);
   OUT=t(:,i);
   ALLIN=[ALLIN IN];
   ALLOUT=[ALLOUT OUT];
   [r,N]=size(ALLIN);
   [r,s]=size(CRBF);
   dd=mdist(IN,CRBF,width);
   md=sqrt(dd);
   [d,ind]=min(md);
   % Caculating the actual output of ith sample data
   ai=exp(-dd);
   ai1=transf(ai,IN);
   ai2=w2*ai1;
   errout=t(:,i)-ai2;
   errout1=errout.*errout;
   errout2=sum(errout1)/s2;
   e(i)=sqrt(errout2);
   if i<=q/3
      ke=emax;
      kd=kdmax;
      ks=0.8;
   elseif i>q/3 & i<2*q/3 
      ke=max(emax*beta.^(i-q/3),emin);
      kd=max(kdmax*gama.^(i-q/3),kdmin);
      ks=0.9;
   else
      ke=emin;
      kd=kdmin;
      ks=0.95;
   end   
   if d > kd
      if e(i) > ke
         %compute widths and centers
         extc=[CRBF range];
         extw=[width width0];
         diffc=extc-IN*ones(1,s+2);
         for J=1:r
            cvar=abs(diffc(J,:));
            [cdmin,nmin]=min(cvar);
            if cdmin<=km
               CRBF(J,s+1)=extc(J,nmin);
               width(J,s+1)=extw(J,nmin);
            else
               CRBF(J,s+1)=IN(J);
               width(J,s+1)=k*cdmin;
            end
         end
    
         % Compare the CRBF,delect the repeating ones
         oldCRBF=CRBF;
         oldwidth=width;
         CCRBF=[];
         newwd=[];
         [r,s1]=size(CRBF);
         while s1>1
            redc=CRBF;
            redc(:,1)=[];
            for j2=1:s1-1
               dcc=CRBF(:,1)-redc(:,j2);
               if all(dcc==0) break, end              
            end              
            if any(dcc~=0)
               CCRBF=[CCRBF CRBF(:,1)];
               newwd=[newwd width(:,1)];
            end   
            CRBF(:,1)=[];
            width(:,1)=[];
            s1=s1-1;   
         end
         CCRBF=[CCRBF CRBF(:,1)];
         newwd=[newwd width(:,1)];
         CRBF=CCRBF;
         width=newwd;
         w1=CRBF';
         [u,r]=size(w1);
         % Caculating outputs of RBF after growing for all coming data
         A=exp(-mdist(ALLIN,CRBF,width));
         A0=transf(A,ALLIN);
         if u*(r+1)<=N
            % caculating error reduction rate
            err=cperr(A0,ALLOUT);  %err is s2*u(r+1), which corresponds to w2
            errT=err';
            err1=zeros(u,s2*(r+1));
            err1(:)=errT;          % err1 is u*s2(r+1)         
            err21=err1';  
            % err21 ,s2(r+1)*u,corresponds to w21, in which every element
            % means the importance of the correspond weight coefficient in
            % w21. The bigger, the more important.
            err22=sum(err21.*err21)/(s2*(r+1));
            err23=sqrt(err22);
            No=find(err23<kerr);
            if ~isempty(No)
               delc=CRBF(:,No);
               delw=width(:,No);
               CRBF(:,No)=[];
               w1(No,:)=[];
               width(:,No)=[];
               rCRBF=CRBF;
               rwidth=width;
               err21(:,No)=[];
               [u1,r]=size(w1);
               A=exp(-mdist(ALLIN,CRBF,width));
               A0=transf(A,ALLIN);           
               w2=ALLOUT/A0;         
               A1=w2*A0;
               derr=ALLOUT-A1;
               sse=sumsqr(derr)/(i*s2);
               rmse=sqrt(sse);
               RMSE(i)=rmse;
               rule(i)=u1;
            else
               w2=ALLOUT/A0;   % w2 is s2*u(r+1)
               A1=w2*A0;
               derr=ALLOUT-A1;
               sse=sumsqr(derr)/(s2*i);
               rmse=sqrt(sse);
               RMSE(i)=rmse;
               rule(i)=u;
            end
         else 
            w2=ALLOUT/A0;   % w2 is s2*u(r+1)
            A1=w2*A0;
            derr=ALLOUT-A1;
            sse=sumsqr(derr)/(s2*i);
            rmse=sqrt(sse);
            RMSE(i)=rmse;
            rule(i)=u;
         end
      else
         [w2,A1,A]=mweight(ALLIN,CRBF,width,ALLOUT);
         derr=ALLOUT-A1;
         sse=sumsqr(derr)/(i*s2);
         rmse=sqrt(sse);
         RMSE(i)=rmse;
         rule(i)=s;      
      end
   else
      if e(i) > ke  
         if s*(r+1)<=N
            aa2=exp(-mdist(ALLIN,CRBF,width));
            aa3=transf(aa2,ALLIN);
            werr=cperr(aa3,ALLOUT);  %err is s2*u(r+1), which corresponds to w2
            werrT=werr';
            werr1=zeros(s,s2*(r+1));
            werr1(:)=werrT;          % err1 corresponds to ww2,which is u*s2(r+1)         
            werr21=werr1';         
            err31=reshape(werr21(:,ind),r+1,s2);
            err32=abs(err31');
            if s2>1
               err33=sum(err32); %err33(j+1) represents the total ERR for input variable j                
            else
               err33=err32;
            end
            err3=sum(err33); % err3 represents the total ERR for ind-th rule 
            err30=err33(1);
            rate0=err30/err3;
            err33(1)=[];
            err3r=err33;
            rater=err3r./(err3-err30);
            nm=find(rater<=1/r);
            width(nm,ind)=ks.*width(nm,ind);
            [w2,A1,A]=mweight(ALLIN,CRBF,width,ALLOUT);
            derr=ALLOUT-A1;
            sse=sumsqr(derr)/(i*s2);
            rmse=sqrt(sse);
            RMSE(i)=rmse;
            rule(i)=s;
         else
            width(:,ind)=ks.*width(:,ind);
            [w2,A1,A]=mweight(ALLIN,CRBF,width,ALLOUT);
            derr=ALLOUT-A1;
            sse=sumsqr(derr)/(i*s2);
            rmse=sqrt(sse);
            RMSE(i)=rmse;
            rule(i)=s;
         end
      else
         [w2,A1,A]=mweight(ALLIN,CRBF,width,ALLOUT);
         derr=ALLOUT-A1;
         sse=sumsqr(derr)/(i*s2);
         rmse=sqrt(sse);
         RMSE(i)=rmse;
         rule(i)=s;
      end
   end
end          