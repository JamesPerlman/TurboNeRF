cmake_minimum_required(VERSION 3.24)
project(glad LANGUAGES C)

set(SOURCES
    src/glad.c
)

set(HEADERS
    include/glad/glad.h
    include/KHR/khrplatform.h
)

include_directories(include)

add_library(glad STATIC ${SOURCES} ${HEADERS})
