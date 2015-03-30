.onLoad = function(lib, pkg) 
{
    init()
}

.onAttach = function(lib, pkg) 
{
    mess = paste("RcppGP version: ", utils::packageDescription("RcppGP")$Version, "\n", sep="")

    packageStartupMessage(mess, appendLF = TRUE)
}

.onUnload <- function(libpath) 
{
    finalize()
}