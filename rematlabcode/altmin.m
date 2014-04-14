%  X    d x n
%  X =  Wopt * Zopt 
%  Z    k x n
%  Y    m x n
%  Y =  S * X
%  S    m x d
%  W    d x k

rnds = 10;
lambda = 0.3;
k = 1000;
m = 100;
d = size(X,1);
n = size(X,2);

W = randn(d,k);
for j=1:d
    W(j,:) = W(j,:)/norm(W(j,:));
end

Wav = W;
for i = 1:30
    for t = 1:n
        eta = 1/(2*sqrt(n));
        S = randn(m,d)/sqrt(d);
        Y = S*X(:,t);
        Z = com_sen(Y,S*W,lambda,k,rnds);
        W = W - eta*(W*Z - X(:,t))*Z';
        %for j=1:d
        %    W(j,:) = W(j,:)/norm(W(j,:));
        %end
        NN = norm(W);
        if(NN > 16)
            W = W*16/NN;
        end
    end
    i
end
