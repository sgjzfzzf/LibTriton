#!/bin/bash

find examples trident-core python test -name "*.py" | xargs ruff format
find trident-core -name "*.c" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" -o -name "*.td" | xargs clang-format -i
(find trident-core -name CMakeLists.txt && echo CMakeLists.txt) | xargs cmake-format -i
