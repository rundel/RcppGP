n1 = 10
n2 = 5

pts1 = matrix(runif(2*n1),ncol=2)
pts2 = matrix(runif(2*n2),ncol=2)

d1 = as.matrix( dist(pts1) )
d2 = as.matrix( dist(pts2) )
d3 = as.matrix( dist(rbind(pts1,pts2)) )[1:n1,n1+1:n2]

nug_cov = function(d, p) 
{   
    tauSq = p[1]
    r = d
    r[] = 0
    r[d==0] = tauSq

    return(r)
}

exp_cov = function(d, p)
{
    sigmaSq = p[1]
    phi = p[2]

    return( sigmaSq * exp(-d*phi) )
}

const_cov = function(d, p)
{
    C = p[1]
    r = d
    r[] = C
    
    return(r)
}