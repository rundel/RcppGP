AC_INIT([RcppGP], [0.1.0], [rundel@gmail.com])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR(config)

AM_INIT_AUTOMAKE

AC_PROG_LIBTOOL

#################### GCC Compiler ####################

AC_PROG_CXX

#################### R stuff ####################

AC_ARG_WITH([r-include],
    AC_HELP_STRING([--with-r-include=r_include],
        [specify the full path to your R header files, e.g. /usr/share/R/include]),
    [R_INCLUDE="$withval"])

AC_ARG_WITH([r-lib],
    AC_HELP_STRING([--with-r-lib=r-lib],
        [specify the full path to your R shared libraries, e.g. /usr/lib64/R/lib]),
    [R_LIB="$withval"])

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
    AC_MSG_ERROR("could not determine R_HOME")
else
    CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
    if test -z "${R_INCLUDE}"; then
        R_INCLUDE=`"${R_HOME}/bin/R" CMD config --cppflags`
    else
        R_INCLUDE="-I${R_INCLUDE}"
    fi

    if test -z "${R_LIB}"; then
        R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
    else
        R_LIB="-L${R_LIB} -lR"
    fi

    RCPPARMA_FLAGS=`${R_HOME}/bin/Rscript -e "RcppArmadillo:::CxxFlags()"`
    RCPP_FLAGS=`${R_HOME}/bin/Rscript -e "Rcpp:::CxxFlags()"`
    RCPP_LIBS=`${R_HOME}/bin/Rscript -e "Rcpp:::LdFlags()"`

    R_BLAS=`"${R_HOME}/bin/R" CMD config BLAS_LIBS`
    R_LAPACK=`"${R_HOME}/bin/R" CMD config LAPACK_LIBS`

    R_CPPFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`

    CPPFLAGS+=" ${R_INCLUDE} ${R_CPPFLAGS} ${RCPP_FLAGS} ${RCPPARMA_FLAGS}"
    LIBS+=" ${R_LIB} ${RCPP_LIBS} ${R_BLAS} ${R_LAPACK}"
fi

#################### Boost Stuff ####################

AC_ARG_WITH(
    [boost-home],
    [AC_HELP_STRING([--with-boost-home=DIR],[full path to BOOST installation @<:@/usr/local@:>@])],
    [BOOST_HOME="$withval"],
    [BOOST_HOME=/usr/local]
)
AC_MSG_RESULT([setting BOOST home directory to ${BOOST_HOME}])

BOOST_INC=${BOOST_HOME}/include
AC_MSG_CHECKING([for BOOST headers])
AC_CHECK_FILE(
    [${BOOST_INC}/boost/timer.hpp],
    [BOOST_CPPFLAGS="-I${BOOST_INC}"],
    [AC_MSG_ERROR([Unable to find BOOST timer.hpp header])]
)
CPPFLAGS+=" ${BOOST_CPPFLAGS}"

LDFLAGS+=" -lboost_timer -lboost_system"

#################### Defaults     ####################

USE_MAGMA=yes

#################### CUDA Toolkit ####################

AC_ARG_WITH(
    [cuda-home],
    [AC_HELP_STRING([--with-cuda-home=DIR],[full path to CUDA installation @<:@/usr/local/cuda@:>@])],
    [CUDA_HOME="$withval"],
    [CUDA_HOME=/usr/local/cuda]
)
AC_MSG_RESULT([setting CUDA home directory to ${CUDA_HOME}])

NVCC=${CUDA_HOME}/bin/nvcc
NCFLAGS="-arch=sm_20 -O2 -Xcompiler \"-O2 ${R_CPICFLAGS}\""
AC_CHECK_FILE(
    [${NVCC}],
    ,
    [USE_MAGMA=no]
)

CUDA_INC=${CUDA_HOME}/include
AC_MSG_CHECKING([for CUDA headers])
AC_CHECK_FILE(
    [${CUDA_INC}/cublas.h],
    [CUDA_CPPFLAGS="-I${CUDA_INC}"],
    [USE_MAGMA=no]
)

CUDA_LIBS="-lcudart -lcublas -lcusolver -lcurand"
AC_MSG_CHECKING([whether this is a 64-bit version of CUDA])
AC_CHECK_FILE(
    [${CUDA_HOME}/lib64/libcublas.so],
    [CUDA_LDFLAGS="-L${CUDA_HOME}/lib64"],
    [CUDA_LDFLAGS="-L${CUDA_HOME}/lib"]
)

#################### MAGMA Libraries ####################

AC_ARG_WITH([magma-home],
    [AC_HELP_STRING([--with-magma-home=DIR],[full path to MAGMA installion directory @<:@/usr/local/magma@:>@])],
    [MAGMA_HOME="$withval"],
    [MAGMA_HOME=/usr/local/magma]
)

MAGMA_LIB=${MAGMA_HOME}/lib
MAGMA_INC=${MAGMA_HOME}/include

AC_MSG_CHECKING([for MAGMA headers])
AC_CHECK_FILE(
    [${MAGMA_INC}/magma.h],
    [],
    [USE_MAGMA=no]
)

AC_MSG_CHECKING([for MAGMA static library])
AC_CHECK_FILE(
    [${MAGMA_LIB}/libmagma.a],
    [],
    [USE_MAGMA=no]
)

if test "${USE_MAGMA}" == "yes"; then
    CPPFLAGS+=" ${CUDA_CPPFLAGS} -DHAVE_CUBLAS -DUSE_GPU -I${MAGMA_INC}"
    LDFLAGS+=" ${CUDA_LDFLAGS} -L${MAGMA_LIB}"
    LIBS+=" ${CUDA_LIBS} -lmagma"
    CU_FILES="\$(wildcard *.cu)"
    NCFLAGS="-arch=sm_20 -O2 -Xcompiler \"-O2 -fPIC\""

else
    AC_MSG_RESULT([Unable to locate magma library / headers, compiling without gpu support])
fi

CPPFLAGS+=" -fPIC"

#AC_MSG_RESULT([CPPFLAGS ${CPPFLAGS}])
#AC_MSG_RESULT([LIBS ${LIBS}])
#AC_MSG_RESULT([LDFLAGS ${LDFLAGS}])


#################### Makevars ####################

AC_SUBST([CPPFLAGS])
AC_SUBST([LDFLAGS])
AC_SUBST([LIBS])
AC_SUBST([CU_FILES])
AC_SUBST([NVCC])
AC_SUBST([NCFLAGS])

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
