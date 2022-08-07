#ifndef PIVOT_UTILS_H
#define PIVOT_UTILS_H

#ifndef NDEBUG
#define assertSuccess(err)                                                    \
    {                                                                         \
        if (err != MPI_SUCCESS) {                                             \
            char errStr[100];                                                 \
            int strLen;                                                       \
            MPI_Error_string(err, errStr, &strLen);                           \
            printf("Err 0x%X in line %d : %s\n", int(err), __LINE__, errStr); \
            abort();                                                          \
        }                                                                     \
    }
#else
#define assertSuccess(err)
#endif
#define CHKERR(func)             \
    {                            \
        int _errCode = (func);   \
        assertSuccess(_errCode); \
    }

// constants
#define BUFF 1000

#endif // PIVOT_UTILS_H