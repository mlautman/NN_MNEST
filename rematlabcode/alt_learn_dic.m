%  X    d x n
%  X =  Wopt * Zopt 
%  Z    k x n
%  Y    m x n
%  Y =  S * X
%  S    m x d
%  W    d x k

rnds = 10;
lambda = 0.015;
k = 1000;
m = 100;
d = size(X,1);
n = size(X,2);

W = randn(d,k);
for j=1:d
    W(j,:) = W(j,:)/norm(W(j,:));
end

Wav = W;
for i = 1:100
    for t = 1:n
        eta = 1/(2*sqrt(n));
        S = randn(m,d)/sqrt(m*d);
        Y = S*X(:,t);
        Z(:,t) = com_sen(Y,S*W,lambda,k,rnds);
    end
    W = X*Z';
    for j=1:d
        W(j,:) = W(j,:)/norm(W(j,:));
    end
    i
end
