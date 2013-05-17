#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config.h"
#include "config.h"
#endif

#include "audio/audio.h"

struct audio_playout_buffer;

int audio_playout_buffer_init(struct audio_playout_buffer **);
void audio_playout_buffer_destroy(struct audio_playout_buffer *);
void audio_playout_buffer_write(struct audio_playout_buffer *, struct audio_frame *);
int audio_playout_buffer_read(struct audio_playout_buffer *, char *buffer,
                int samples, int ch_count, int bps);
void audio_playout_buffer_poison(struct audio_playout_buffer *);

