source("settings.R")

context("Testing Combined Covariance Models")

test_that("Nug+Exp Covariance", {
    m1 = build_cov_model(nugget_cov,exponential_cov)
    m2 = build_cov_model(exponential_cov,nugget_cov)

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

test_that("Nug*Exp Covariance", {
    m1 = build_cov_model(nugget_cov,exponential_cov,method="multi")
    m2 = build_cov_model(exponential_cov,nugget_cov,method="multi")

    expect_that( calc_cov(m1, d1, c(1,2,3)), is_equivalent_to( calc_cov(m2, d1, c(2,3,1)) ) )
    expect_that( calc_cov(m1, d2, c(1,2,3)), is_equivalent_to( calc_cov(m2, d2, c(2,3,1)) ) )
    expect_that( calc_cov(m1, d3, c(1,2,3)), is_equivalent_to( calc_cov(m2, d3, c(2,3,1)) ) )  
    
    expect_that( calc_cov(m1, d1, c(1,2,3)), is_equivalent_to(nug_cov(d1, 1)*exp_cov(d1, c(2,3))) )
    expect_that( calc_cov(m1, d2, c(1,2,3)), is_equivalent_to(nug_cov(d2, 1)*exp_cov(d2, c(2,3))) )
    expect_that( calc_cov(m1, d3, c(1,2,3)), is_equivalent_to(nug_cov(d3, 1)*exp_cov(d3, c(2,3))) )  
    
    expect_error(calc_cov(m, d1, c(-1, 2, 3)))
    expect_error(calc_cov(m, d1, c( 1,-2, 3)))
    expect_error(calc_cov(m, d1, c( 1, 2,-3)))
    expect_error(calc_cov(m, d1, c(-1,-2, 3)))

    expect_error(calc_cov(m1, d1, 1))
    expect_error(calc_cov(m1, d1, 1:2))
    expect_error(calc_cov(m1, d1, 1:4))
})
