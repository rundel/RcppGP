posdef = function (n, ev = runif(n, 0, 10)) 
{
    Z = matrix(ncol=n, rnorm(n^2))
    decomp = qr(Z)
    Q = qr.Q(decomp) 
    R = qr.R(decomp)
    d = diag(R) 
    O = Q %*% diag(d / abs(d))
    Z = t(O) %*% diag(ev) %*% O
    return(Z)
}

context("Testing Cholesky Update and Downdate")

P = posdef(2000)

for(n in c(10,100,1000))#,2500))
{
    test_that(paste0("Testing with n=",n), {
        n_up_err = 0
        n_down_err = 0
        
        for(rep in 1:5)
        {
            s = sample(1:nrow(P), n)
            M = P[s,s]

            L = t(chol(M))
            v = 0.01 * matrix(runif(n),nrow=n)
            vv = v %*% t(v)

            up_chol = try(t(chol(M + vv)), TRUE)
            down_chol = try(t(chol(M - vv)), TRUE)

            if (class(up_chol) != "try-error") {
                up = .Call("chol_update_test",L,v,PACKAGE="RcppGP")
                expect_true(abs(sum(up-up_chol)) < 1e-9)
            } else {
                expect_error(.Call("chol_update_test",L,v,PACKAGE="RcppGP"))
                n_up_err = n_up_err + 1
            }

            if (class(down_chol) != "try-error") {
                down = .Call("chol_downdate_test",L,v,PACKAGE="RcppGP")
                expect_true(abs(sum(down-down_chol)) < 1e-9)
            } else {
                expect_error(.Call("chol_downdate_test",L,v,PACKAGE="RcppGP"))
                n_up_err = n_up_err + 1
            }
        }
        expect_true(n_up_err != 5)
        expect_true(n_down_err != 5)
    })
}

