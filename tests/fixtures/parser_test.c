#include <stdint.h>
#include <stddef.h>

int parse_test_case(const uint8_t *data, size_t len) {
    return len > 0 && data[0] == 0;
}
