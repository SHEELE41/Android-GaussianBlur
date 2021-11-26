# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("opencl")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library(GLES_mali STATIC IMPORTED)

add_library(
        OpenCLDriver
        SHARED
        OpenCLDriver.c)

find_library(log-lib log)

find_library(jnigraphics-lib jnigraphics )

target_link_libraries( # Specifies the target library.
        OpenCLDriver
        # "C:/Users/parlab/AppData/Local/Android/Sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include/CL/OpenCL_lib_and_include/include/CL/OpenCL_lib_and_include/lib/libGLES_mali.so"
        # C:/Users/parlab/AppData/Local/Android/Sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/include/CL/OpenCL_lib_and_include/include/CL/OpenCL_lib_and_include/lib/libGLES_mali.so
        # Links the target library to the log library
        # included in the NDK.
        # "C:\\Users\\parlab\\AppData\\Local\\Android\\Sdk\\ndk\\21.4.7075529\\toolchains\\llvm\\prebuilt\\windows-x86_64\\sysroot\\usr\\include\\CL\\OpenCL_lib_and_include\\lib\\libGLES_mali.so"
        C:/Users/parlab/Documents/OpenCL_lib_and_include/lib/libGLES_mali.so
        ${log-lib} ${jnigraphics-lib})

target_include_directories(OpenCLDriver PRIVATE src/main/cpp)