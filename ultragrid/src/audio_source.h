#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_win32.h"
#include "config_unix.h"
#endif /* HAVE_CONFIG_H */

#define AUDIO_SOURCE_PROPERTY_FPS 0 /* double */

#ifdef __cplusplus
extern "C" {
#endif

struct audio_source;
struct audio_frame;

struct audio_source     *audio_source_init(char *filename);
struct audio_frame      *audio_source_read(struct audio_source *, int frame_number);
void                     audio_source_destroy(struct audio_source *state);
int                      audio_source_set_property(struct audio_source *state, int property, void *val, size_t len);

#ifdef __cplusplus
}
#endif

