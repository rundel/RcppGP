nugget_cov = list(type = "nugget", params = list( 
                list( name = "tauSq",
                      dist = "IG",
                      trans = "log",
                      start = 1,
                      tuning = 0.01,
                      hyperparams = c(0.1, 0.1)
                    )
             ))

constant_cov = list(type = "constant", params = list( 
                    list( name = "C",
                          dist = "IG",
                          trans = "log",
                          start = 1,
                          tuning = 0.01,
                          hyperparams = c(0.1, 0.1)
                        )
               ))

exponential_cov = list(type = "exponential", params = list(
                        list( name = "sigmaSq",
                              dist = "ig",
                              trans = "log",
                              start = 1,
                              tuning = 0.01,
                              hyperparams = c(0.1, 0.1)
                            ),         
                        list( name = "phi",
                              dist = "Unif",
                              trans = "logit",
                              start = 1,
                              tuning = 0.01,
                              hyperparams = c(0, 100)
                            )            
                  ))