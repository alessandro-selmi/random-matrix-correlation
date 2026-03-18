function C = gaussian_mix_cdf(x,mu,s)

    N = length(mu);
    
    C = 0;
    
    for i = 1:N
       
        C = C + 1 + erf((x-mu(i))/(sqrt(2)*s(i)));
        
    end
    
    C = C/2;

end