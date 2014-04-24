function V = Gama(U,lambda,X,W)

k = length(U);
V = U + W'*(X - W*U);

for i = 1 : k
    if(abs(V(i)) > lambda)
        V(i) = V(i) - lambda*sign(V(i));
    else
        V(i) = 0;
    end
end
