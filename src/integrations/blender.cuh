#pragma once

#include "../common.h"
#include <glfw/glfw3.h>

NRC_NAMESPACE_BEGIN

void test() {
    glfwInit();
    glfwTerminate();
    printf("The test has completed.");
}

NRC_NAMESPACE_END
