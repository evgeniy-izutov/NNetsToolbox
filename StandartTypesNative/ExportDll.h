#ifdef STANDARDTYPESAPI
#define STANDARDTYPES_EXPORT __declspec(dllexport)
#else
#define STANDARDTYPES_EXPORT __declspec(dllimport)
#endif