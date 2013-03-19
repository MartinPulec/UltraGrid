#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_win32.h"
#include "config_unix.h"
#endif // HAVE_CONFIG_H

void (*exit_uv)(int status) = NULL;
volatile int should_exit = FALSE;
int uv_argc;
char **uv_argv;
long packet_rate = 13600;
uint32_t RTT = 0;               /* this is computed by randle_rr in rtp_callback */
volatile int logo_hidden = 0;
const char * volatile video_directory = 0;

