set(SOURCE_FILES
    "rasterize.cu"
    "rasterize.h"
    "rasterizeTools.h"
    )


add_library(src ${SOURCE_FILES})
set_target_properties(src PROPERTIES LINKER_LANGUAGE CUDA)