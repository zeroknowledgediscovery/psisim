#ifndef DEBUG_MACROS_H
#define DEBUG_MACROS_H

#include <iostream>

using namespace std;

#ifdef DEBUG_MODE
  #define DEBUG_PRINT(x) do { cout << x << endl; } while (0)
  #define DEBUG_VAR(name, value) do { cout << "[DEBUG] " << name << " = " << value << endl; } while (0)
#else
  #define DEBUG_PRINT(x) do {} while (0)
  #define DEBUG_VAR(name, value) do {} while (0)
#endif

#endif // DEBUG_MACROS_H
