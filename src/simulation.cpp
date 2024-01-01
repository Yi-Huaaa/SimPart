
#include "simulation.hpp"


// Gates: NOT, AND, OR, NOR, XOR
/* todo: modify gates to multiple bits */

int NOT (int b) {
    return ~b;
}

int AND (int b1, int b2) {
    return b1&b2;
}

int OR (int b1, int b2) {
    return b1|b2;
}

int NOR (int b1, int b2) {
    return ~(b1|b2);
}

int XOR (int b1, int b2) {
    return b1^b2;
}