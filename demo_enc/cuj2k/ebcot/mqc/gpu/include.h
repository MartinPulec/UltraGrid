#include "common.h"

#ifdef MQC_GPU_CODENAME
    #undef MQC_GPU_CODENAME
#endif

#ifdef MQC_GPU_NAME
    #undef MQC_GPU_NAME
#endif

#ifdef MQC_GPU_FINAL
    #include "final.h"
    #define MQC_GPU_CODENAME final
    #define MQC_GPU_NAME "Final"
    #undef MQC_GPU_FINAL
#endif

#ifdef MQC_GPU_DEVELOP
    #include "develop.h"
    #define MQC_GPU_CODENAME develop
    #define MQC_GPU_NAME "Develop"
    #undef MQC_GPU_DEVELOP
#endif

#ifdef MQC_GPU_ORIGINAL
    #include "original.h"
    #define MQC_GPU_CODENAME original
    #define MQC_GPU_NAME "Original"
    #undef MQC_GPU_ORIGINAL
#endif

#ifdef MQC_GPU_IMPR_RENORM
    #include "impr_renorm.h"
    #define MQC_GPU_CODENAME impr_renorm
    #define MQC_GPU_NAME "Renormalize CLZ"
    #undef MQC_GPU_IMPR_RENORM
#endif

#ifdef MQC_GPU_IMPR_RENORM_TABLE
    #include "impr_renorm_table.h"
    #define MQC_GPU_CODENAME impr_renorm_table
    #define MQC_GPU_NAME "Renormalize LT"
    #undef MQC_GPU_IMPR_RENORM_TABLE
#endif

#ifdef MQC_GPU_LOOP_UNROLLING
    #include "loop_unrolling.h"
    #define MQC_GPU_CODENAME loop_unrolling
    #define MQC_GPU_NAME "Loop Unrolling"
    #undef MQC_GPU_LOOP_UNROLLING
#endif

#ifdef MQC_GPU_PREFIX_SUM
    #include "prefix_sum.h"
    #define MQC_GPU_CODENAME prefix_sum
    #define MQC_GPU_NAME "Prefix Sum"
    #undef MQC_GPU_PREFIX_SUM
#endif

#ifdef MQC_GPU_SHARED
    #include "shared.h"
    #define MQC_GPU_CODENAME shared
    #define MQC_GPU_NAME "Shared Memory"
    #undef MQC_GPU_SHARED
#endif

#ifdef MQC_GPU_REGISTER
    #include "register.h"
    #define MQC_GPU_CODENAME register
    #define MQC_GPU_NAME "Register Memory"
    #undef MQC_GPU_REGISTER
#endif

#ifdef MQC_GPU_NAME
    #define MQC_GPU_DECLARATION_BUILD(MQC_GPU_CODENAME,NAME)   mqc_gpu_ ## MQC_GPU_CODENAME ## NAME
    #define MQC_GPU_DECLARATION_EXPAND(MQC_GPU_CODENAME,NAME)  MQC_GPU_DECLARATION_BUILD(MQC_GPU_CODENAME,NAME)
    #define MQC_GPU_DECLARATION(NAME)                          MQC_GPU_DECLARATION_EXPAND(MQC_GPU_CODENAME,NAME)

    #define mqc_gpu_create  MQC_GPU_DECLARATION(_create)
    #define mqc_gpu_encode  MQC_GPU_DECLARATION(_encode)
    #define mqc_gpu_destroy MQC_GPU_DECLARATION(_destroy)
#endif

