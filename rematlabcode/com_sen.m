function Z = com_sen(Y,M,lambda,k,rnds)

Zp = ones(k,1);
Z = Zp;
d = length(Y);
eta = 1;

for t = 1:rnds
    Zpp = Z;
    etap = eta;
    eta = (1 + sqrt(1 + 4*etap^2))/2;
    Z = Gama(Z + ((etap - 1)/eta)*(Z - Zp),lambda,Y,M);
    Zp = Zpp;
end

