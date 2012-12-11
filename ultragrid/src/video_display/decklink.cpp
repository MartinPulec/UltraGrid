/*
 * FILE:    video_display/decklink.cpp
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          Colin Perkins    <csp@isi.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 * Copyright (c) 2001-2003 University of Southern California
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "host.h"
#include "debug.h"
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "video_codec.h"
#include "tv.h"
#include "video_display/decklink.h"
#include "debug.h"
#include "video_capture.h"
#include "audio/audio.h"
#include "audio/utils.h"

#include "DeckLinkAPI.h"
#include "DeckLinkAPIVersion.h"

#ifdef __cplusplus
} // END of extern "C"
#endif

#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <string>
#include <utility>

// defined int video_capture/decklink.cpp
void print_output_modes(IDeckLink *);

#ifdef HAVE_MACOSX
#define STRING CFStringRef
#else
#define STRING const char *
#endif

#define MAX_DEVICES 4

#define MAX_QUEUE_LEN 10

using namespace std;

class PlaybackDelegate : public IDeckLinkVideoOutputCallback // , public IDeckLinkAudioOutputCallback
{
        struct state_decklink *                 s;
        int                                     i;

public:
        PlaybackDelegate (struct state_decklink* owner, int index);

        // IUnknown needs only a dummy implementation
        virtual HRESULT         QueryInterface (REFIID iid, LPVOID *ppv)        {return E_NOINTERFACE;}
        virtual ULONG           AddRef ()                                                                       {return 1;}
        virtual ULONG           Release ()                                                                      {return 1;}

        virtual HRESULT         ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result);
        virtual HRESULT         ScheduledPlaybackHasStopped ();
        //virtual HRESULT         RenderAudioSamples (bool preroll);
};


class DeckLinkTimecode : public IDeckLinkTimecode{
                BMDTimecodeBCD timecode;
        public:
                DeckLinkTimecode() : timecode(0) {}
                /* IDeckLinkTimecode */
                virtual BMDTimecodeBCD GetBCD (void) { return timecode; }
                virtual HRESULT GetComponents (/* out */ uint8_t *hours, /* out */ uint8_t *minutes, /* out */ uint8_t *seconds, /* out */ uint8_t *frames) { 
                        *frames =   (timecode & 0xf)              + ((timecode & 0xf0) >> 4) * 10;
                        *seconds = ((timecode & 0xf00) >> 8)      + ((timecode & 0xf000) >> 12) * 10;
                        *minutes = ((timecode & 0xf0000) >> 16)   + ((timecode & 0xf00000) >> 20) * 10;
                        *hours =   ((timecode & 0xf000000) >> 24) + ((timecode & 0xf0000000) >> 28) * 10;
                        return S_OK;
                }
                virtual HRESULT GetString (/* out */ STRING *timecode) { return E_FAIL; }
                virtual BMDTimecodeFlags GetFlags (void)        { return bmdTimecodeFlagDefault; }
                virtual HRESULT GetTimecodeUserBits (/* out */ BMDTimecodeUserBits *userBits) { if (!userBits) return E_POINTER; else return S_OK; }

                /* IUnknown */
                virtual HRESULT         QueryInterface (REFIID iid, LPVOID *ppv)        {return E_NOINTERFACE;}
                virtual ULONG           AddRef ()                                                                       {return 1;}
                virtual ULONG           Release ()                                                                      {return 1;}
                
                void SetBCD(BMDTimecodeBCD timecode) { this->timecode = timecode; }
        };

class DeckLinkFrame;
class DeckLinkFrame : public IDeckLinkMutableVideoFrame
{
                long ref;
                
                long width;
                long height;
                long rawBytes;
                BMDPixelFormat pixelFormat;
                char *data;

                IDeckLinkTimecode *timecode;

        protected:
                DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf); 
                virtual ~DeckLinkFrame();

        public:
        	static DeckLinkFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat);                
                /* IUnknown */
                virtual HRESULT QueryInterface(REFIID, void**);
                virtual ULONG AddRef();
                virtual ULONG Release();
                
                /* IDeckLinkVideoFrame */
                long GetWidth (void);
                long GetHeight (void);
                long GetRowBytes (void);
                BMDPixelFormat GetPixelFormat (void);
                BMDFrameFlags GetFlags (void);
                HRESULT GetBytes (/* out */ void **buffer);
                
                HRESULT GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode);
                HRESULT GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary);
                

                /* IDeckLinkMutableVideoFrame */
                HRESULT SetFlags(BMDFrameFlags);
                HRESULT SetTimecode(BMDTimecodeFormat, IDeckLinkTimecode*);
                HRESULT SetTimecodeFromComponents(BMDTimecodeFormat, uint8_t, uint8_t, uint8_t, uint8_t, BMDTimecodeFlags);
                HRESULT SetAncillaryData(IDeckLinkVideoFrameAncillary*);
                HRESULT SetTimecodeUserBits(BMDTimecodeFormat, BMDTimecodeUserBits);
};

class DeckLink3DFrame : public DeckLinkFrame, public IDeckLinkVideoFrame3DExtensions
{
        private:
                DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf); 
                ~DeckLink3DFrame();
                
                long ref;
                
                long width;
                long height;
                long rawBytes;
                BMDPixelFormat pixelFormat;
                DeckLinkFrame *rightEye;

        public:
                static DeckLink3DFrame *Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat);
                
                /* IUnknown */
                HRESULT QueryInterface(REFIID, void**);
                ULONG AddRef();
                ULONG Release();

                /* IDeckLinkVideoFrame3DExtensions */
                BMDVideo3DPackingFormat Get3DPackingFormat();
                HRESULT GetFrameForRightEye(IDeckLinkVideoFrame**);
};

#define DECKLINK_MAGIC DISPLAY_DECKLINK_ID

struct device_state {
        PlaybackDelegate        *delegate;
        IDeckLink               *deckLink;
        IDeckLinkOutput         *deckLinkOutput;
        IDeckLinkMutableVideoFrame *deckLinkFrame;
        //IDeckLinkVideoFrame *deckLinkFrame;
};

struct state_decklink {
        state_decklink() : 
                        timecode(0), play_audio(0) {
                pthread_mutex_init(&lock, NULL);
                pthread_cond_init(&in_cv, NULL);
                pthread_cond_init(&out_cv, NULL);

                memset(&audio, 0, sizeof(audio));
                memset(&desc, 0, sizeof(desc));
        }
        virtual ~state_decklink() {
                pthread_mutex_destroy(&lock);
                pthread_cond_destroy(&in_cv);
                pthread_cond_destroy(&out_cv);
        }

        pthread_t           worker_id;

        uint32_t            magic;

        struct timeval      tv;

        struct device_state state[MAX_DEVICES];

        BMDTimeValue        frameRateDuration;
        BMDTimeScale        frameRateScale;

        DeckLinkTimecode    *timecode;

        struct audio_frame  audio;
        struct video_desc   desc;

        unsigned long int   frames;
        unsigned long int   frames_last;
        bool                stereo;
        bool                initialized;
        bool                emit_timecode;
        int                 devices_cnt;
        unsigned int        play_audio:1;
        int                 output_audio_channel_count;
        BMDPixelFormat                    pixelFormat;

        int                 dev_x, dev_y;

        pthread_mutex_t     lock;
        pthread_cond_t      in_cv;
        pthread_cond_t      out_cv;
        queue<struct video_frame *> frame_queue;
 };

static void update_timecode(DeckLinkTimecode *tc, double fps);
static void *worker(void *args);

static void *worker(void *args) {
        struct state_decklink *s = (struct state_decklink *) args;

        while(1) {
                struct video_frame *frame;
                pthread_mutex_lock(&s->lock);
                while(s->frame_queue.empty()) {
                        pthread_cond_wait(&s->in_cv, &s->lock);
                }
                frame = s->frame_queue.front();
                s->frame_queue.pop();
                pthread_cond_signal(&s->out_cv);
                pthread_mutex_unlock(&s->lock);

                if(!frame) {
                        // received poisoned pill
                        break;
                }

                uint32_t i;
                for(int i = 0; i < s->devices_cnt; ++i) {
                        s->state[i].deckLinkFrame = DeckLinkFrame::Create(s->desc.width / s->dev_x,
                                        s->desc.height / s->dev_y,
                                        vc_get_linesize(s->desc.width, s->desc.color_spec) / s->dev_x, s->pixelFormat);
                }

                for(int x = 0; x < s->dev_x; ++x) {
                        for(int y = 0; y < s->dev_y; ++y) {
                                char *data;
                                s->state[x + y * s->dev_x].deckLinkFrame->GetBytes((void **) &data);
                                int untiled_linesize = vc_get_linesize(s->desc.width, s->desc.color_spec);
                                for (int line = y * (s->desc.height / s->dev_y);
                                                line < (y + 1) * (s->desc.height / s->dev_y);
                                                ++line) {
                                        int linesize = untiled_linesize / s->dev_x;
                                        //char *src = data + line * s->tile->linesize;
                                        char *src = frame->tiles[0].data + line * untiled_linesize;
                                        src += x * linesize;
                                        memcpy(data, src, linesize);
                                        data += linesize;
                                }
                        }
                }

                vf_free_data(frame);

                for (int j = 0; j < s->devices_cnt; ++j) {
                        if(s->emit_timecode) {
                                s->state[j].deckLinkFrame->SetTimecode(bmdVideoOutputRP188, s->timecode);
                        }
                        s->state[j].deckLinkOutput->ScheduleVideoFrame(s->state[j].deckLinkFrame,
                                        s->frames * s->frameRateDuration, s->frameRateDuration, s->frameRateScale);
                }

                if(s->emit_timecode) {
                        update_timecode(s->timecode, s->desc.fps);
                }
        }

        return NULL;
}


static struct display_device *get_devices();

static struct video_desc *get_mode_list(IDeckLink *deckLink, ssize_t *count)
{	
        *count = -1;
        struct video_desc *ret;

        IDeckLinkOutput *deckLinkOutput;
        if (deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&deckLinkOutput) != S_OK) {
                if(deckLinkOutput != NULL)
                        deckLinkOutput->Release();
                return NULL;
        }

        IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;
        BMDDisplayMode			  displayMode = bmdModeUnknown;

        
        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator)))
        {
                fprintf(stderr, "Fatal: cannot create display mode iterator [decklink].\n");
                return NULL;
        }

        ret = NULL;
        *count = 0;

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                int curMode = *count;
                *count += 1;

                ret = (struct video_desc *) realloc(ret, *count * sizeof(struct video_desc));

                ret[curMode].width = deckLinkDisplayMode->GetWidth();
                ret[curMode].height = deckLinkDisplayMode->GetHeight();
                BMDTimeValue        frameRateDuration;
                BMDTimeScale        frameRateScale;
                deckLinkDisplayMode->GetFrameRate(&frameRateDuration,
                                &frameRateScale);
                ret[curMode].fps = (double) frameRateScale / frameRateDuration;
                BMDFieldDominance dominance = deckLinkDisplayMode->GetFieldDominance();
                switch(dominance) {
                        case bmdLowerFieldFirst:
                        case bmdUpperFieldFirst:
                                ret[curMode].interlacing = INTERLACED_MERGED;
                                break;
                        case bmdProgressiveFrame:
                                ret[curMode].interlacing = PROGRESSIVE;
                                break;
                        case bmdProgressiveSegmentedFrame:
                                ret[curMode].interlacing = SEGMENTED_FRAME;
                                break;
                        default:
                                cerr << "[Decklink] Unknown mode encountered!!!!" << endl;
                                ret[curMode].interlacing = PROGRESSIVE; // whatever it is
                                // should be skipped
                }
                ret[curMode].tile_count = 0;
        }
        displayModeIterator->Release();

        deckLinkOutput->Release();
        
        return ret;

}

struct groupped_item_data {
        list<int> devices;
        struct video_desc       *modes;
        // -1 means that driver doesnt provide modes
        ssize_t                  modes_count;
};

bool has_same_set_of_modes(struct video_desc *set1, ssize_t set1_size,
                struct video_desc *set2, ssize_t set2_size);
// ordered set for now is sufficient (Decklink Quad)
bool has_same_set_of_modes(struct video_desc *set1, ssize_t set1_size,
                struct video_desc *set2, ssize_t set2_size)
{
        if(set1_size != set2_size) {
                return false;
        }

        for (int i = 0; i < set1_size; ++i) {
                if(set1[i].width !=  set2[i].width ||
                                set1[i].height != set2[i].height ||
                                fabs(set1[i].fps - set1[i].fps) >= 0.01 ||
                                set1[i].interlacing != set2[i].interlacing) {
                        return false;
                }
        }

        return true;
}

void multiply_mode_sizes_by(struct video_desc *set, ssize_t set_size,
                size_t x, size_t y);
void multiply_mode_sizes_by(struct video_desc *set, ssize_t set_size,
                size_t x, size_t y)
{
        for (int i = 0; i < set_size; ++i) {
                set[i].width *= x;
                set[i].height *= y;
        }
}

static struct display_device *get_devices(void)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;
        int                             numDevices = 0;
        HRESULT                         result;

        typedef map<string, groupped_item_data > group_map;
        group_map                       groups; // device prefix, indices

        struct display_device *ret = (struct display_device *) malloc(sizeof(struct display_device) * (numDevices + 1));

        ret[numDevices].name = NULL;

        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        deckLinkIterator = CreateDeckLinkIteratorInstance();
        if (deckLinkIterator == NULL)
        {
		fprintf(stderr, "\nA DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
                return ret;
        }
        
        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                STRING          deviceNameString = NULL;
                const char     *name;

                ret = (struct display_device *) realloc((void *) ret, sizeof(struct display_device) * (numDevices + 2)); // current count + this + EOR
                
                // *** Print the model name of the DeckLink card
                result = deckLink->GetDisplayName((STRING *) &deviceNameString);
#ifdef HAVE_MACOSX
                name = ret[numDevices].name = (char *) malloc(128);
                CFStringGetCString(deviceNameString, (char *) ret[numDevices].name, 128, kCFStringEncodingMacRoman);
#else
                name = ret[numDevices].name = deviceNameString;
#endif

                ssize_t count = -1;
                ret[numDevices].modes = get_mode_list(deckLink, &count);
                ret[numDevices].modes_count = count;

                if(strchr(ret[numDevices].name, '(')) {
                        size_t len = strchr(ret[numDevices].name, '(') - ret[numDevices].name;
                        group_map::iterator it = groups.find(string(name, len));
                        if(it == groups.end() ||
                                        !has_same_set_of_modes(it->second.modes, it->second.modes_count,
                                                ret[numDevices].modes, ret[numDevices].modes_count)) {
                                groupped_item_data data;
                                data.devices.push_front(numDevices);
                                if(count >= 0) {
                                        data.modes = (struct video_desc *) malloc(sizeof(struct video_desc) *
                                                        count);
                                        memcpy(data.modes, ret[numDevices].modes, count *
                                                        sizeof(struct video_desc));
                                        data.modes_count = count;
                                }

                                pair<string, groupped_item_data > new_item(string(name, len),
                                                        data);
                                groups.insert(new_item);
#if 0
                                pair<string, list<int> > new_item(string(name, len),
                                                        list<int>());
                                new_item.second.push_front(numDevices);
                                groups.insert(new_item);
#endif
                        } else {
                                it->second.devices.push_back(numDevices);
                        }
                }

                if (result == S_OK)
                {
                        char *tmp = (char *) malloc(128);
                        snprintf(tmp, 128, "decklink:%u", (unsigned int) numDevices);
                        ret[numDevices].driver_identifier = tmp;

#ifdef HAVE_MACOSX
                        CFRelease(deviceNameString);
#endif
                }
                
                // Increment the total number of DeckLink cards found
                numDevices++;
        
                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }
        
        deckLinkIterator->Release();

        for(group_map::iterator it = groups.begin(); it != groups.end(); ++it) {
                // tiled 4K
                if(it->second.devices.size() >= 4) {
                        ret = (struct display_device *) realloc((void *) ret, sizeof(struct display_device) * (numDevices + 2)); // current count + this + EOR
                        ret[numDevices].name = (char *) malloc(it->first.size() + strlen("(groupped)") + 1);
                        strcpy((char *) ret[numDevices].name, it->first.c_str());
                        strcat((char *) ret[numDevices].name, "(groupped)");

                        ret[numDevices].driver_identifier = (char *) malloc(128); 
                        strcpy((char *) ret[numDevices].driver_identifier, "decklink:");

                        size_t devices_included_in_group = 0;
                        for(list<int>::iterator device_index = it->second.devices.begin();
                                        device_index != it->second.devices.end();
                                        ++device_index) {
                                devices_included_in_group += 1;
                                char index_str[8];
                                snprintf(index_str, sizeof(index_str), "%s%d",
                                                (device_index != it->second.devices.begin() ? "," : ""),
                                                *device_index);
                                strcat((char *) ret[numDevices].driver_identifier, index_str);
                                // we do not more than 4 devices in this branch
                                if(devices_included_in_group == 4)
                                        break;
                        }

                        ret[numDevices].modes = it->second.modes;
                        ret[numDevices].modes_count = it->second.modes_count;
                        // adjust modes to 2x2 grid
                        multiply_mode_sizes_by(ret[numDevices].modes, ret[numDevices].modes_count,
                                        2, 2);

                        numDevices++;
                }
        }

        ret[numDevices].name = NULL;

        return ret;
}

static void show_help(void);

static void show_help(void)
{
        IDeckLinkIterator*              deckLinkIterator;
        IDeckLink*                      deckLink;
        int                             numDevices = 0;
                HRESULT                         result;

        printf("Decklink (output) options:\n");
        printf("\t-d decklink:<device_numbers>[:3D][:timecode] - coma-separated numbers of output devices\n");
        
        // Create an IDeckLinkIterator object to enumerate all DeckLink cards in the system
        deckLinkIterator = CreateDeckLinkIteratorInstance();
        if (deckLinkIterator == NULL)
        {
		fprintf(stderr, "\nA DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
                return;
        }
        
        // Enumerate all cards in this system
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                STRING          deviceNameString = NULL;
                const char *deviceNameCString;
                
                // *** Print the model name of the DeckLink card
                result = deckLink->GetModelName((STRING *) &deviceNameString);
#ifdef HAVE_MACOSX
                deviceNameCString = (char *) malloc(128);
                CFStringGetCString(deviceNameString, (char *) deviceNameCString, 128, kCFStringEncodingMacRoman);
#else
                deviceNameCString = deviceNameString;
#endif
                if (result == S_OK)
                {
                        printf("\ndevice: %d.) %s \n\n",numDevices, deviceNameCString);
                        print_output_modes(deckLink);
#ifdef HAVE_MACOSX
                        CFRelease(deviceNameString);
#endif
                        free((void *)deviceNameCString);
                }
                
                // Increment the total number of DeckLink cards found
                numDevices++;
        
                // Release the IDeckLink instance when we've finished with it to prevent leaks
                deckLink->Release();
        }
        
        deckLinkIterator->Release();

        // If no DeckLink cards were found in the system, inform the user
        if (numDevices == 0)
        {
                printf("\nNo Blackmagic Design devices were found.\n");
                return;
        } 
}


struct video_frame *
display_decklink_getf(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;
        struct video_frame *ret = NULL;

        assert(s->magic == DECKLINK_MAGIC);

        if (s->initialized) {
                if(s->stereo) {
#if 0
                        assert(s->devices_cnt == 0);
                        s->state[0].deckLinkFrame = DeckLink3DFrame::Create(s->frame->tiles[0].width, s->frame->tiles[0].height,
                                                s->frame->tiles[0].linesize, s->pixelFormat);
                                
                        s->state[0].deckLinkFrame->GetBytes((void **) &s->frame->tiles[0].data);
                        IDeckLinkVideoFrame *right;
                        
                        dynamic_cast<DeckLink3DFrame *>(s->state[0].deckLinkFrame)->GetFrameForRightEye(&right);
                        right->GetBytes((void **) &s->frame->tiles[1].data);
#endif
                } else {
                        ret = vf_alloc_desc_data(s->desc);
                }
        }

        return ret;
}

static void update_timecode(DeckLinkTimecode *tc, double fps)
{
        const float epsilon = 0.005;
        int shifted;
        uint8_t hours, minutes, seconds, frames;
        BMDTimecodeBCD bcd;
        bool dropFrame = false;

        if(ceil(fps) - fps > epsilon) { /* NTSCi drop framecode  */
                dropFrame = true;
        }

        tc->GetComponents (&hours, &minutes, &seconds, &frames);
        frames++;

        if((double) frames > fps - epsilon) {
                frames = 0;
                seconds++;
                if(seconds >= 60) {
                        seconds = 0;
                        minutes++;
                        if(dropFrame) {
                                if(minutes % 10 != 0)
                                        seconds = 2;
                        }
                        if(minutes >= 60) {
                                minutes = 0;
                                hours++;
                                if(hours >= 24) {
                                        hours = 0;
                                }
                        }
                }
        }

        bcd = (frames % 10) | (frames / 10) << 4 | (seconds % 10) << 8 | (seconds / 10) << 12 | (minutes % 10)  << 16 | (minutes / 10) << 20 |
                (hours % 10) << 24 | (hours / 10) << 28;

        tc->SetBCD(bcd);
}

int display_decklink_putf(void *state, char *frame)
{
        int tmp;
        struct state_decklink *s = (struct state_decklink *)state;
        struct timeval tv;

        UNUSED(frame);

        assert(s->magic == DECKLINK_MAGIC);


#if 0
        uint32_t i;
        s->state[0].deckLinkOutput->GetBufferedVideoFrameCount(&i);
        //if (i > 2) 
        if (0) 
                fprintf(stderr, "Frame dropped!\n");
        else {
                for(int x = 0; x < s->dev_x; ++x) {
                        for(int y = 0; y < s->dev_y; ++y) {
                                char *data;
                                s->state[x + y * s->dev_x].deckLinkFrame->GetBytes((void **) &data);
                                for (int line = y * (s->tile->height / s->dev_y);
                                                line < (y + 1) * (s->tile->height / s->dev_y);
                                                ++line) {
                                        int linesize = s->tile->linesize / s->dev_x;
                                        //char *src = data + line * s->tile->linesize;
                                        char *src = s->frame->tiles[0].data + line * s->tile->linesize;
                                        src += x * linesize;
                                        memcpy(data, src, linesize);
                                        data += linesize;
                                }
                        }
                }

                for (int j = 0; j < s->devices_cnt; ++j) {
                        if(s->emit_timecode) {
                                s->state[j].deckLinkFrame->SetTimecode(bmdVideoOutputRP188, s->timecode);
                        }
                        s->state[j].deckLinkOutput->ScheduleVideoFrame(s->state[j].deckLinkFrame,
                                        s->frames * s->frameRateDuration, s->frameRateDuration, s->frameRateScale);
                }

                s->frames++;
                if(s->emit_timecode) {
                        update_timecode(s->timecode, s->frame->fps);
                }
        }
#endif
        pthread_mutex_lock(&s->lock);
        while(s->frame_queue.size() > MAX_QUEUE_LEN) {
                pthread_cond_wait(&s->out_cv, &s->lock);
        }

        s->frame_queue.push((struct video_frame *) frame);
        pthread_cond_signal(&s->in_cv);
        pthread_mutex_unlock(&s->lock);

        s->frames++;


        gettimeofday(&tv, NULL);
        double seconds = tv_diff(tv, s->tv);
        if (seconds > 5) {
                double fps = (s->frames - s->frames_last) / seconds;
                fprintf(stdout, "%lu frames in %g seconds = %g FPS\n",
                        s->frames - s->frames_last, seconds, fps);
                s->tv = tv;
                s->frames_last = s->frames;
        }

        return TRUE;
}

static BMDDisplayMode get_mode(IDeckLinkOutput *deckLinkOutput, struct video_desc desc, BMDTimeValue *frameRateDuration,
		BMDTimeScale        *frameRateScale, int index)
{	IDeckLinkDisplayModeIterator     *displayModeIterator;
        IDeckLinkDisplayMode*             deckLinkDisplayMode;
        BMDDisplayMode			  displayMode = bmdModeUnknown;
        
        // Populate the display mode combo with a list of display modes supported by the installed DeckLink card
        if (FAILED(deckLinkOutput->GetDisplayModeIterator(&displayModeIterator)))
        {
                fprintf(stderr, "Fatal: cannot create display mode iterator [decklink].\n");
                return (BMDDisplayMode) -1;
        }

        while (displayModeIterator->Next(&deckLinkDisplayMode) == S_OK)
        {
                STRING modeNameString;
                const char *modeNameCString;
                if (deckLinkDisplayMode->GetName(&modeNameString) == S_OK)
                {
#ifdef HAVE_MACOSX
                        modeNameCString = (char *) malloc(128);
                        CFStringGetCString(modeNameString, (char *) modeNameCString, 128, kCFStringEncodingMacRoman);
#else
                        modeNameCString = modeNameString;
#endif
                        if (deckLinkDisplayMode->GetWidth() == desc.width &&
                                        deckLinkDisplayMode->GetHeight() == desc.height)
                        {
                                double displayFPS;
                                BMDFieldDominance dominance;
                                bool interlaced;

                                dominance = deckLinkDisplayMode->GetFieldDominance();
                                if (dominance == bmdLowerFieldFirst ||
                                                dominance == bmdUpperFieldFirst)
                                        interlaced = true;
                                else // progressive, psf, unknown
                                        interlaced = false;

                                deckLinkDisplayMode->GetFrameRate(frameRateDuration,
                                                frameRateScale);
                                displayFPS = (double) *frameRateScale / *frameRateDuration;
                                if(fabs(desc.fps - displayFPS) < 0.01 && (desc.interlacing == INTERLACED_MERGED ? interlaced : !interlaced)
                                  )
                                {
                                        printf("Device %d - selected mode: %s\n", index, modeNameCString);
                                        displayMode = deckLinkDisplayMode->GetDisplayMode();
                                        break;
                                }
                        }
                }
        }
        displayModeIterator->Release();
        
        return displayMode;
}

int
display_decklink_reconfigure(void *state, struct video_desc desc)
{
        struct state_decklink            *s = (struct state_decklink *)state;
        
        bool                              modeFound = false;
        BMDDisplayMode                    displayMode;
        BMDDisplayModeSupport             supported;
        int h_align = 0;

        assert(s->magic == DECKLINK_MAGIC);

        pthread_mutex_lock(&s->lock);
        while(!s->frame_queue.empty()) {
                pthread_cond_wait(&s->out_cv, &s->lock);
        }
        
        s->desc = desc;

	switch (desc.color_spec) {
                case UYVY:
                        s->pixelFormat = bmdFormat8BitYUV;
                        break;
                case v210:
                        s->pixelFormat = bmdFormat10BitYUV;
                        break;
                case RGBA:
                        s->pixelFormat = bmdFormat8BitBGRA;
                        break;
                default:
                        fprintf(stderr, "[DeckLink] Unsupported pixel format!\n");
        }

	if(s->stereo) {
#if 0
		for (int i = 0; i < 2; ++i) {
			struct tile  *tile = vf_get_tile(s->frame, i);
			tile->width = desc.width;
		        tile->height = desc.height;
	                tile->linesize = vc_get_linesize(tile->width, s->frame->color_spec);
	                tile->data_len = tile->linesize * tile->height;
	        }
		displayMode = get_mode(s->state[0].deckLinkOutput, desc, &s->frameRateDuration,
                                                &s->frameRateScale, 0);
                if(displayMode == (BMDDisplayMode) -1)
                        goto error;
		
		s->state[0].deckLinkOutput->DoesSupportVideoMode(displayMode, s->pixelFormat, bmdVideoOutputDualStream3D,
	                                &supported, NULL);
                if(supported == bmdDisplayModeNotSupported)
                {
                        fprintf(stderr, "[decklink] Requested parameters combination not supported - %dx%d@%f.\n", desc.width, desc.height, (double)desc.fps);
                        goto error;
                }
                

                s->state[0].deckLinkOutput->EnableVideoOutput(displayMode,  bmdVideoOutputDualStream3D);
                s->state[0].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration);
#endif
        } else {
                if(desc.tile_count > s->devices_cnt) {
                        fprintf(stderr, "[decklink] Expected at most %d streams. Got %d.\n", s->devices_cnt,
                                        desc.tile_count);
                        goto error;
                }

                s->dev_x = s->dev_y = 1;

                if(s->devices_cnt == 4) {
                        s->dev_x = s->dev_y = 2;
                }

	        for(int i = 0; i < s->devices_cnt; ++i) {
                        BMDVideoOutputFlags outputFlags= bmdVideoOutputFlagDefault;
	                
                        struct video_desc device_desc = desc;
                        device_desc.width /= s->dev_x;
                        device_desc.height /= s->dev_y;
	                displayMode = get_mode(s->state[i].deckLinkOutput, device_desc, &s->frameRateDuration,
                                                &s->frameRateScale, i);
                        if(displayMode == (BMDDisplayMode) -1)
                                goto error;

                        if(s->emit_timecode) {
                                outputFlags = bmdVideoOutputRP188;
                        }
	
	                s->state[i].deckLinkOutput->DoesSupportVideoMode(displayMode, s->pixelFormat, outputFlags,
	                                &supported, NULL);
	                if(supported == bmdDisplayModeNotSupported)
	                {
                                fprintf(stderr, "[decklink] Requested parameters "
                                                "combination not supported - %d * %dx%d@%f, timecode %s.\n",
                                                desc.tile_count, desc.width, desc.height, desc.fps,
                                                (outputFlags & bmdVideoOutputRP188 ? "ON": "OFF"));
	                        goto error;
	                }
	
	                s->state[i].deckLinkOutput->EnableVideoOutput(displayMode, outputFlags);
	        }
	
	        for(int i = 0; i < s->devices_cnt; ++i) {
	                s->state[i].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, (double) s->frameRateDuration);
	        }
	}

        s->initialized = true;
        pthread_mutex_unlock(&s->lock);
        return TRUE;

error:
        pthread_mutex_unlock(&s->lock);
        return FALSE;
}


void *display_decklink_init(char *fmt, unsigned int flags)
{
        struct state_decklink *s;
        IDeckLinkIterator*                              deckLinkIterator;
        HRESULT                                         result;
        int                                             cardIdx[MAX_DEVICES];
        int                                             dnum = 0;

        s = new state_decklink;
        s->magic = DECKLINK_MAGIC;
        s->stereo = FALSE;
        s->emit_timecode = false;
        
        if(fmt == NULL) {
                cardIdx[0] = 0;
                s->devices_cnt = 1;
                fprintf(stderr, "Card number unset, using first found (see -d decklink:help)!\n");

        } else if (strcmp(fmt, "help") == 0) {
                show_help();
                return NULL;
        } else  {
                char *tmp = strdup(fmt);
                char *ptr;
                char *saveptr1 = 0ul, *saveptr2 = 0ul;

                ptr = strtok_r(tmp, ":", &saveptr1);                
                char *devices = strdup(ptr);
                s->devices_cnt = 0;
                ptr = strtok_r(devices, ",", &saveptr2);
                do {
                        cardIdx[s->devices_cnt] = atoi(ptr);
                        ++s->devices_cnt;
                } while ((ptr = strtok_r(NULL, ",", &saveptr2)));
                free(devices);
                
                ptr = strtok_r(NULL, ":", &saveptr1);
                if(ptr) {
                        if(strcasecmp(ptr, "3D") == 0) {
                                s->stereo = true;
                                ptr = strtok_r(NULL, ":", &saveptr1);
                                if(strcasecmp(ptr, "timecode") == 0) {
                                        s->emit_timecode = true;
                                }
                        } else if(strcasecmp(ptr, "timecode") == 0) {
                                s->emit_timecode = true;
                        } else {
                                fprintf(stderr, "[DeckLink] Warning: unknown options in config string.\n");
                        }
                }
                free (tmp);
        }
	assert(!s->stereo || s->devices_cnt == 1);

        gettimeofday(&s->tv, NULL);

        // Initialize the DeckLink API
        deckLinkIterator = CreateDeckLinkIteratorInstance();
        if (!deckLinkIterator)
        {
		fprintf(stderr, "\nA DeckLink iterator could not be created. The DeckLink drivers may not be installed or are outdated.\n");
		fprintf(stderr, "This UltraGrid version was compiled with DeckLink drivers %s. You should have at least this version.\n\n",
                                BLACKMAGIC_DECKLINK_API_VERSION_STRING);
                return NULL;
        }

        for(int i = 0; i < s->devices_cnt; ++i) {
                s->state[i].deckLink = NULL;
                s->state[i].deckLinkOutput = NULL;
        }

        // Connect to the first DeckLink instance
        IDeckLink    *deckLink;
        while (deckLinkIterator->Next(&deckLink) == S_OK)
        {
                bool found = false;
                for(int i = 0; i < s->devices_cnt; ++i) {
                        if (dnum == cardIdx[i]){
                                s->state[i].deckLink = deckLink;
                                found = true;
                        }
                }
                if(!found && deckLink != NULL)
                        deckLink->Release();
                dnum++;
        }
        for(int i = 0; i < s->devices_cnt; ++i) {
                if(s->state[i].deckLink == NULL) {
                        fprintf(stderr, "No DeckLink PCI card #%d found\n", cardIdx[i]);
                        return NULL;
                }
        }

        if(flags & DISPLAY_FLAG_ENABLE_AUDIO) {
                s->play_audio = TRUE;
                s->audio.data = NULL;
        } else {
                s->play_audio = FALSE;
        }
        
        if(s->emit_timecode) {
                s->timecode = new DeckLinkTimecode;
        } else {
                s->timecode = NULL;
        }
        
        for(int i = 0; i < s->devices_cnt; ++i) {
                // Obtain the audio/video output interface (IDeckLinkOutput)
                if (s->state[i].deckLink->QueryInterface(IID_IDeckLinkOutput, (void**)&s->state[i].deckLinkOutput) != S_OK) {
                        if(s->state[i].deckLinkOutput != NULL)
                                s->state[i].deckLinkOutput->Release();
                        s->state[i].deckLink->Release();
                        return NULL;
                }

                s->state[i].delegate = new PlaybackDelegate(s, i);
                // Provide this class as a delegate to the audio and video output interfaces
                s->state[i].deckLinkOutput->SetScheduledFrameCompletionCallback(s->state[i].delegate);
                //s->state[i].deckLinkOutput->DisableAudioOutput();
        }

        s->frames = 0;
        s->initialized = false;

        if(pthread_create(&s->worker_id, NULL, worker, (void*)s) != 0) {
                fprintf(stderr, "[Decklink] Error creating thread!!!!");
                return NULL;
        }

        return (void *)s;
}

void display_decklink_run(void *state)
{
        UNUSED(state);
}

void display_decklink_finish(void *state)
{
        UNUSED(state);
}

void display_decklink_done(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;

        // NULL is poisoned pill
        display_decklink_putf(s, NULL);
        pthread_join(s->worker_id, NULL);

        delete s->timecode;
        delete s;
}

display_type_t *display_decklink_probe(void)
{
        display_type_t *dtype;

        dtype = (display_type_t *) malloc(sizeof(display_type_t));
        if (dtype != NULL) {
                dtype->id = DISPLAY_DECKLINK_ID;
                dtype->name = "decklink";
                dtype->description = "Blackmagick DeckLink card";

                dtype->devices = get_devices();
        }
        return dtype;
}

int display_decklink_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decklink *s = (struct state_decklink *)state;
        codec_t codecs[] = {v210, UYVY, RGBA};
        
        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RSHIFT:
                        *(int *) val = 16;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_GSHIFT:
                        *(int *) val = 8;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BSHIFT:
                        *(int *) val = 0;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        if(s->devices_cnt == 1 && !s->stereo)
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_MERGED;
                        else
                                *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        break;

                default:
                        return FALSE;
        }
        return TRUE;
}

PlaybackDelegate::PlaybackDelegate (struct state_decklink * owner, int index) 
        : s(owner), i(index)
{
}

HRESULT         PlaybackDelegate::ScheduledFrameCompleted (IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result) 
{
	completedFrame->Release();
        return S_OK;
}

HRESULT         PlaybackDelegate::ScheduledPlaybackHasStopped ()
{
        return S_OK;
}

/*
 * AUDIO
 */
struct audio_frame * display_decklink_get_audio_frame(void *state)
{
        struct state_decklink *s = (struct state_decklink *)state;
        
        if(!s->play_audio)
                return NULL;
        return &s->audio;
}

void display_decklink_put_audio_frame(void *state, struct audio_frame *frame)
{
        struct state_decklink *s = (struct state_decklink *)state;
        unsigned int sampleFrameCount = frame->data_len / (s->audio.bps *
                        s->audio.ch_count);
        unsigned int sampleFramesWritten;

        /* we got probably count that cannot be played directly (probably 1) */
        if(s->output_audio_channel_count != s->audio.ch_count) {
                assert(s->audio.ch_count == 1); /* only reasonable value so far */
                if (sampleFrameCount * s->output_audio_channel_count 
                                * frame->bps > frame->max_size) {
                        fprintf(stderr, "[decklink] audio buffer overflow!\n");
                        sampleFrameCount = frame->max_size / 
                                        (s->output_audio_channel_count * frame->bps);
                        frame->data_len = sampleFrameCount *
                                        (frame->ch_count * frame->bps);
                }
                
                audio_frame_multiply_channel(frame,
                                s->output_audio_channel_count);
        }
        
	s->state[0].deckLinkOutput->ScheduleAudioSamples (frame->data, sampleFrameCount, 0,
                0, &sampleFramesWritten);
        if(sampleFramesWritten != sampleFrameCount)
                fprintf(stderr, "[decklink] audio buffer underflow!\n");

}

void display_decklink_audio_reset(void *state)
{
        UNUSED(state);
}

int display_decklink_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate) {
        struct state_decklink *s = (struct state_decklink *)state;
        BMDAudioSampleType sample_type;

        if(s->audio.data != NULL)
                free(s->audio.data);
                
        s->audio.bps = quant_samples / 8;
        s->audio.sample_rate = sample_rate;
        s->output_audio_channel_count = s->audio.ch_count = channels;
        
        if (s->audio.ch_count != 1 &&
                        s->audio.ch_count != 2 && s->audio.ch_count != 8 &&
                        s->audio.ch_count != 16) {
                fprintf(stderr, "[decklink] requested channel count isn't supported: "
                        "%d\n", s->audio.ch_count);
                s->play_audio = FALSE;
                return FALSE;
        }
        
        /* toggle one channel to supported two */
        if(s->audio.ch_count == 1) {
                 s->output_audio_channel_count = 2;
        }
        
        if((quant_samples != 16 && quant_samples != 32) ||
                        sample_rate != 48000) {
                fprintf(stderr, "[decklink] audio format isn't supported: "
                        "samples: %d, sample rate: %d\n",
                        quant_samples, sample_rate);
                s->play_audio = FALSE;
                return FALSE;
        }
        switch(quant_samples) {
                case 16:
                        sample_type = bmdAudioSampleType16bitInteger;
                        break;
                case 32:
                        sample_type = bmdAudioSampleType32bitInteger;
                        break;
                default:
                        return FALSE;
        }
                        
        s->state[0].deckLinkOutput->EnableAudioOutput(bmdAudioSampleRate48kHz,
                        sample_type,
                        s->output_audio_channel_count,
                        bmdAudioOutputStreamContinuous);
        s->state[0].deckLinkOutput->StartScheduledPlayback(0, s->frameRateScale, s->frameRateDuration);
        
        s->audio.max_size = 5 * (quant_samples / 8) 
                        * s->audio.ch_count
                        * sample_rate;                
        s->audio.data = (char *) malloc (s->audio.max_size);

        return TRUE;
}

bool operator==(const REFIID & first, const REFIID & second){
    return (first.byte0 == second.byte0) &&
        (first.byte1 == second.byte1) &&
        (first.byte2 == second.byte2) &&
        (first.byte3 == second.byte3) &&
        (first.byte4 == second.byte4) &&
        (first.byte5 == second.byte5) &&
        (first.byte6 == second.byte6) &&
        (first.byte7 == second.byte7) &&
        (first.byte8 == second.byte8) &&
        (first.byte9 == second.byte9) &&
        (first.byte10 == second.byte10) &&
        (first.byte11 == second.byte11) &&
        (first.byte12 == second.byte12) &&
        (first.byte13 == second.byte13) &&
        (first.byte14 == second.byte14) &&
        (first.byte15 == second.byte15);
}

HRESULT DeckLinkFrame::QueryInterface(REFIID id, void**frame)
{
        return E_NOINTERFACE;
}

ULONG DeckLinkFrame::AddRef()
{
        return ++ref;
}

ULONG DeckLinkFrame::Release()
{
        if(--ref == 0)
                delete this;
	return ref;
}

DeckLinkFrame::DeckLinkFrame(long w, long h, long rb, BMDPixelFormat pf)
	: width(w), height(h), rawBytes(rb), pixelFormat(pf), ref(1l)
{
        data = new char[rb * h];
        timecode = NULL;
}

DeckLinkFrame *DeckLinkFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat)
{
        return new DeckLinkFrame(width, height, rawBytes, pixelFormat);
}


DeckLinkFrame::~DeckLinkFrame() 
{
	delete[] data;
}

long DeckLinkFrame::GetWidth ()
{
        return width;
}

long DeckLinkFrame::GetHeight ()
{
        return height;
}

long DeckLinkFrame::GetRowBytes ()
{
        return rawBytes;
}

BMDPixelFormat DeckLinkFrame::GetPixelFormat ()
{
        return pixelFormat;
}

BMDFrameFlags DeckLinkFrame::GetFlags ()
{
        return bmdFrameFlagDefault;
}

HRESULT DeckLinkFrame::GetBytes (/* out */ void **buffer)
{
        *buffer = static_cast<void *>(data);
        return S_OK;
}

HRESULT DeckLinkFrame::GetTimecode (/* in */ BMDTimecodeFormat format, /* out */ IDeckLinkTimecode **timecode)
{
        *timecode = dynamic_cast<IDeckLinkTimecode *>(this->timecode);
        return S_OK;
}

HRESULT DeckLinkFrame::GetAncillaryData (/* out */ IDeckLinkVideoFrameAncillary **ancillary)
{
	return S_FALSE;
}

/* IDeckLinkMutableVideoFrame */
HRESULT DeckLinkFrame::SetFlags (/* in */ BMDFrameFlags newFlags)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetTimecode (/* in */ BMDTimecodeFormat format, /* in */ IDeckLinkTimecode *timecode)
{
        if(this->timecode)
                this->timecode->Release();
        this->timecode = timecode;
        return S_OK;
}

HRESULT DeckLinkFrame::SetTimecodeFromComponents (/* in */ BMDTimecodeFormat format, /* in */ uint8_t hours, /* in */ uint8_t minutes, /* in */ uint8_t seconds, /* in */ uint8_t frames, /* in */ BMDTimecodeFlags flags)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetAncillaryData (/* in */ IDeckLinkVideoFrameAncillary *ancillary)
{
        return E_FAIL;
}

HRESULT DeckLinkFrame::SetTimecodeUserBits (/* in */ BMDTimecodeFormat format, /* in */ BMDTimecodeUserBits userBits)
{
        return E_FAIL;
}



DeckLink3DFrame::DeckLink3DFrame(long w, long h, long rb, BMDPixelFormat pf) 
        : DeckLinkFrame(w, h, rb, pf), ref(1l)
{
        rightEye = DeckLinkFrame::Create(w, h, rb, pf);        
}

DeckLink3DFrame *DeckLink3DFrame::Create(long width, long height, long rawBytes, BMDPixelFormat pixelFormat)
{
        DeckLink3DFrame *frame = new DeckLink3DFrame(width, height, rawBytes, pixelFormat);
        return frame;
}

DeckLink3DFrame::~DeckLink3DFrame()
{
	rightEye->Release();
}

ULONG DeckLink3DFrame::AddRef()
{
        return ++ref;
}

ULONG DeckLink3DFrame::Release()
{
        if(--ref == 0)
                delete this;
	return ref;
}

HRESULT DeckLink3DFrame::QueryInterface(REFIID id, void**frame)
{
        HRESULT result = E_NOINTERFACE;

        if(id == IID_IDeckLinkVideoFrame3DExtensions)
        {
                this->AddRef();
                *frame = dynamic_cast<IDeckLinkVideoFrame3DExtensions *>(this);
                result = S_OK;
        }
        return result;
}


BMDVideo3DPackingFormat DeckLink3DFrame::Get3DPackingFormat()
{
        return bmdVideo3DPackingLeftOnly;
}

HRESULT DeckLink3DFrame::GetFrameForRightEye(IDeckLinkVideoFrame ** frame) 
{
        *frame = rightEye;
        rightEye->AddRef();
        return S_OK;
}
