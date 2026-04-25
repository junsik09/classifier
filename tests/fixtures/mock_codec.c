#include <stdint.h>
#include <stddef.h>

int decode_mock_codec(const uint8_t *data, size_t len) {
    return len > 0 && data[0] == 1;
}
