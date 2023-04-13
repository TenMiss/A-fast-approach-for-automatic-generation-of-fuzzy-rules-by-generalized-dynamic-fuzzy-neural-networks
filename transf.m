function BB=transf(A,P)
% This is the program to generate a matrice in order to caculate
%      consequent parameters
%A---output of RBF
%P-- sample of input
% Revised 11-3-2006
% Copyright Wu Shiqian.
[u,N]=size(A);
[r,N]=size(P);
for j=1:N
   for i=1:r
      PA((i-1)*u+1:i*u,j)=P(i,j)*A(:,j);
   end 
end
BB=[A;PA];
