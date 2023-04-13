function y=mdist(p,c,b)
% This program is used to compute Mahalanobis distance
% p--r*q matrix,which is the incoming sample data
% c--r*s matrix, which is the center matrix
% b--r*s matrix, which is the width matrix
% Revised 11-5-2006
% Copyright Wu Shiqian.
[r,q]=size(p);
[r,s]=size(c);
[r,s]=size(b);
y=zeros(s,q);
if r==1
   for i=1:s
      x=c(:,i)*ones(1,q);
      d=abs(p-x);
      xx=b(:,i)*ones(1,q);
      dd=d./xx;
      y(i,:)=dd.*dd;
   end
else
   for i=1:s
      x=c(:,i)*ones(1,q);
      d=abs(p-x);
      xx=b(:,i)*ones(1,q);
      dd=d./xx;
      y(i,:)=sum(dd.*dd);
   end
end