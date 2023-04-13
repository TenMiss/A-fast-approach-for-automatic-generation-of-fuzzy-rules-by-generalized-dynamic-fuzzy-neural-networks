function y=cperr(a,b)
% This program is used to computes the error reduction ratio
% Revised 11-5-2006
% Copyright Wu Shiqian.
tT=b';
PAT=a';
[WW,AW]=orthogonalize(PAT);
WSSW=sum(WW.*WW)';
WSStT=sum(tT.*tT)';
y=((WW'*tT)'.^2)./(WSStT*WSSW');