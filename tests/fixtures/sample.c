#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

int parse_frame(const uint8_t *data, size_t len) {
    if (len < 4) {
        return -1;
    }
    return memcmp(data, "FRM", 3);
}

static int helper(Context *ctx, const char *path) {
    (void)ctx;
    return path != 0;
}

int decode_message(const char *input, unsigned int input_len, int flags) {
    if (flags) {
        return parse_frame((const uint8_t *)input, input_len);
    }
    return 0;
}

void log_line(FILE *fp, const char *line) {
    fprintf(fp, "%s", line);
}
