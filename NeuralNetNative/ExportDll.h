#ifdef NEURALNETNATIVEAPI
#define NEURALNETNATIVE_EXPORT __declspec(dllexport)
#else
#define NEURALNETNATIVE_EXPORT __declspec(dllimport)
#endif