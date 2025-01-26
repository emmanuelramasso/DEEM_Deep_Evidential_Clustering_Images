function m=labels2matrix(labels, K)

% if ~isempty(find(isnan(labels))), error('Empty labels, exit'); end
% n=length(labels);
% m=zeros(K,n);
% m(labels(:)' + [0:K:K*n-1])=1; 
% m=m';

m=zeros(length(labels),K);
for i=1:length(labels)
    m(i,labels(i))=1;
end
    