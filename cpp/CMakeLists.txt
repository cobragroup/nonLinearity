# Build the C++ shared library
add_library(pmi SHARED pmi.cpp)

# Ensure we build position-independent code
set_target_properties(pmi PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "pmi"
    PREFIX "lib"            # ensures libmylib.so on Unix
    )

# Install it into the Python package _libs folder
install(
    TARGETS pmi
    LIBRARY DESTINATION mip2/_libs   # Linux/macOS
    RUNTIME DESTINATION mip2/_libs   # Windows
    ARCHIVE DESTINATION mip2/_libs   # static if ever built
)
