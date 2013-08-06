.onLoad = function(lib, pkg) 
{
    #.Call("init", PACKAGE="RcppGP")
}

.onAttach = function(lib, pkg) 
{
    mess = paste("RcppGP version: ", utils::packageDescription("RcppGP")$Version, "\n", sep="")

    packageStartupMessage(mess, appendLF = TRUE)
}

.onUnload <- function(libpath) 
{
    #.Call("finalize", PACKAGE="RcppGP")
}