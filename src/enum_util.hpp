#ifndef ENUM_UTIL_HPP
#define ENUM_UTIL_HPP

template<class E> struct enum_map
{
    typedef std::map<std::string, int> map_type;
    static map_type map;
    static std::string name;

    static E from_string(std::string const& str) 
    {
        typename map_type::const_iterator res = map.find(str);
        if (res == map.end())
            throw std::runtime_error("Unknown " + name + " type: " + str + ".");
    
        return static_cast<E>(res->second);
    }

    static std::vector<std::string> valid_keys()
    {
        std::vector<std::string> v;
        typename map_type::const_iterator it = map.begin();
        for(; it != map.end(); ++it)
            v.push_back(it->first);

        return v;
    }
};

template<class P> struct enum_property
{
    typedef std::map<int, P> map_type;
    static map_type map;
    static std::string name;

    static P value(int e) 
    {
        typename map_type::const_iterator res = map.find(e);
        if (res == map.end())
            throw std::runtime_error("Unknown " + name + " type.");
    
        return res->second;
    }
};

#endif