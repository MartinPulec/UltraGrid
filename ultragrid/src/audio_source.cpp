#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "audio_source.h"
#include "audio/audio.h"

#include <sndfile.h>

struct audio_source {
        SNDFILE        *sndfile;

        double          fps;

        struct audio_frame frame;

        int             last_index;
        sf_count_t     last_frames;
};

struct audio_source *audio_source_init(char *filename)
{
        struct audio_source *s;
        SF_INFO         info;

        if(filename == NULL || strcmp(filename, "none") == 0) {
                return NULL;
        }

        s = (struct audio_source *) malloc(sizeof(struct audio_frame));
        if(s == NULL) {
                fprintf(stderr, "Unable to allocate memory.\n");
                return NULL;
        }

        s->sndfile = sf_open(filename, SFM_READ, &info);
        if(s->sndfile == NULL) {
                fprintf(stderr, "Sound file opening error: %s\n", sf_strerror(s->sndfile));
                goto release_state;
        }

        assert(info.seekable);

        s->frame.sample_rate = info.samplerate;
        s->frame.ch_count = info.channels;
        s->fps = 0;

        if(info.format & SF_FORMAT_PCM_24 || info.format & SF_FORMAT_PCM_32 ||
                        info.format & SF_FORMAT_FLOAT ||
                        info.format & SF_FORMAT_DOUBLE) {
                s->frame.bps = 4;
        } else {
                s->frame.bps = 2;
        }

        s->frame.max_size = s->frame.data_len = 0;
        s->frame.data = NULL;

        s->last_index = -1;
        s->last_frames = 0;
                
        return s;

release_state:
        free(s);
        return NULL;
}

struct audio_frame *audio_source_read(struct audio_source *s, int frame_number)
{
        if(s == NULL) {
                return NULL;
        }

        assert(s->fps > 0.0);

        if(s->last_index != frame_number - 1) {
                sf_count_t ret;

                sf_count_t frames = frame_number / s->fps * s->frame.sample_rate;

                ret = sf_seek(s->sndfile, frames, SEEK_SET) ;
                if(ret == -1) {
                        fprintf(stderr, "Audio reading error.\n");
                        return NULL;
                }

                s->last_frames = frame_number / s->fps * s->frame.sample_rate;
        }

        sf_count_t frames = (frame_number + 1) / s->fps * s->frame.sample_rate;
        frames -= s->last_frames;

        sf_count_t ret = 0;

        switch(s->frame.bps) {
                case 2:
                        ret = sf_readf_short(s->sndfile, (short int *) s->frame.data, frames);
                        break;
                case 4:
                        ret = sf_readf_int(s->sndfile, (int *) s->frame.data, frames);
                        break;
        }

        if(ret < frames) {
                printf("End of audio file.\n");
        }

        s->frame.data_len = ret * s->frame.ch_count * s->frame.bps;

        s->last_frames += frames;
        s->last_index = frame_number;
}

void audio_source_destroy(struct audio_source *s)
{
        if(!s) {
                return;
        }

        sf_close(s->sndfile);

        free(s->frame.data);

        free(s);
}

int audio_source_set_property(struct audio_source *s, int property, void *val, size_t len)
{
        if(!s) {
                return TRUE;
        }

        bool ret = FALSE;

        switch(property) {
                case AUDIO_SOURCE_PROPERTY_FPS:
                        assert(len == sizeof(double));
                        s->fps = * (double *) val;
                        s->frame.max_size = s->frame.bps * s->frame.ch_count * s->frame.sample_rate / s->fps * 2; // max 2x frame length
                        free(s->frame.data);
                        s->frame.data = (char *) malloc(s->frame.max_size);
                        ret = TRUE;
                        break;
        }

        return ret;
}

