source("settings.R")

context("Testing Individual Covariance Models")

test_that("Nugget Covariance", {
    m = cov_model(nugget_cov)

    expect_that( calc_cov(m, d1, 0.5), is_equivalent_to(nug_cov(d1, 0.5)) )
    expect_that( calc_cov(m, d2, 0.5), is_equivalent_to(nug_cov(d2, 0.5)) )
    expect_that( calc_cov(m, d3, 0.5), is_equivalent_to(nug_cov(d3, 0.5)) )  

    expect_that( calc_cov(m, d1, 15), is_equivalent_to(nug_cov(d1, 15)) )
    expect_that( calc_cov(m, d2, 15), is_equivalent_to(nug_cov(d2, 15)) )
    expect_that( calc_cov(m, d3, 15), is_equivalent_to(nug_cov(d3, 15)) )  

    expect_error(calc_cov(m, d1, -1))
    expect_error(calc_cov(m, d2, -1))
    expect_error(calc_cov(m, d3, -1))

    expect_error(calc_cov(m, d1, c(1,2)))
})  

test_that("Constant Covariance", {
    m = cov_model(constant_cov)

    expect_that( calc_cov(m, d1, 0.5), is_equivalent_to(const_cov(d1, 0.5)) )
    expect_that( calc_cov(m, d2, 0.5), is_equivalent_to(const_cov(d2, 0.5)) )
    expect_that( calc_cov(m, d3, 0.5), is_equivalent_to(const_cov(d3, 0.5)) )  

    expect_that( calc_cov(m, d1, 15), is_equivalent_to(const_cov(d1, 15)) )
    expect_that( calc_cov(m, d2, 15), is_equivalent_to(const_cov(d2, 15)) )
    expect_that( calc_cov(m, d3, 15), is_equivalent_to(const_cov(d3, 15)) )  

    expect_error(calc_cov(m, d1, -1))
    expect_error(calc_cov(m, d2, -1))
    expect_error(calc_cov(m, d3, -1))

    expect_error(calc_cov(m, d1, c(1,2)))
}) 

test_that("Exponential Covariance", {
    m = cov_model(exponential_cov)

    expect_that( calc_cov(m, d1, c(0.5, 2)), is_equivalent_to(exp_cov(d1, c(0.5, 2))) )
    expect_that( calc_cov(m, d2, c(0.5, 2)), is_equivalent_to(exp_cov(d2, c(0.5, 2))) )
    expect_that( calc_cov(m, d3, c(0.5, 2)), is_equivalent_to(exp_cov(d3, c(0.5, 2))) )  
 
    expect_that( calc_cov(m, d1, c(12, 5)), is_equivalent_to(exp_cov(d1, c(12, 5))) )
    expect_that( calc_cov(m, d2, c(12, 5)), is_equivalent_to(exp_cov(d2, c(12, 5))) )
    expect_that( calc_cov(m, d3, c(12, 5)), is_equivalent_to(exp_cov(d3, c(12, 5))) )  
 
    expect_error(calc_cov(m, d1, c(-1, 2)))
    expect_error(calc_cov(m, d2, c(-1, 2)))
    expect_error(calc_cov(m, d3, c(-1, 2)))
 
    expect_error(calc_cov(m, d1, c(1, -2)))
    expect_error(calc_cov(m, d2, c(1, -2)))
    expect_error(calc_cov(m, d3, c(1, -2)))
 
    expect_error(calc_cov(m, d1, c(-1, -2)))
    expect_error(calc_cov(m, d2, c(-1, -2)))
    expect_error(calc_cov(m, d3, c(-1, -2)))

    expect_error(calc_cov(m, d1, c(1)))
    expect_error(calc_cov(m, d1, c(1,2,3)))
})

test_that("Nug+Exp Covariance", {
    m1 = cov_model(nugget_cov,exponential_cov)
    m2 = cov_model(exponential_cov,nugget_cov)

    expect_that( calc_cov(m1, d1, c(1,2,3)), is_equivalent_to( calc_cov(m2, d1, c(2,3,1)) ) )
    expect_that( calc_cov(m1, d2, c(1,2,3)), is_equivalent_to( calc_cov(m2, d2, c(2,3,1)) ) )
    expect_that( calc_cov(m1, d3, c(1,2,3)), is_equivalent_to( calc_cov(m2, d3, c(2,3,1)) ) )  
    
    expect_that( calc_cov(m1, d1, c(1,2,3)), is_equivalent_to(nug_cov(d1, 1)+exp_cov(d1, c(2,3))) )
    expect_that( calc_cov(m1, d2, c(1,2,3)), is_equivalent_to(nug_cov(d2, 1)+exp_cov(d2, c(2,3))) )
    expect_that( calc_cov(m1, d3, c(1,2,3)), is_equivalent_to(nug_cov(d3, 1)+exp_cov(d3, c(2,3))) )  
    
    expect_error(calc_cov(m, d1, c(-1, 2, 3)))
    expect_error(calc_cov(m, d1, c( 1,-2, 3)))
    expect_error(calc_cov(m, d1, c( 1, 2,-3)))
    expect_error(calc_cov(m, d1, c(-1,-2, 3)))

    expect_error(calc_cov(m1, d1, 1))
    expect_error(calc_cov(m1, d1, 1:2))
    expect_error(calc_cov(m1, d1, 1:4))
})
