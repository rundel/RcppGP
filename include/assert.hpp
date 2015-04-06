#ifndef RT_ASSERT

#include <string>
#include <stdexcept>

#define S(x) #x
#define S_(x) S(x)
#define __SLINE__ S_(__LINE__)

#define RT_ASSERT(cond, msg) if (!(cond)) throw std::runtime_error(std::string(msg) + " (" + __FILE__ + ":" + __SLINE__ +")")

#endif