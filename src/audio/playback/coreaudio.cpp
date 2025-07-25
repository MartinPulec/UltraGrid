/**
 * @file   audio/playback/coreaudio.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2023 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 */
/**
 * @file
 * @todo
 * Remove deprecated instances of ca-disable-adaptive-buf after some transitio period.
 */

#include <AudioUnit/AudioUnit.h>
#include <Availability.h>
#include <MacTypes.h>
#include <cassert>
#include <cstdio>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <CoreAudio/AudioHardware.h>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <string.h>

#include "audio/audio_playback.h"
#include "audio/playback/coreaudio.h"
#include "audio/types.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/audio_buffer.h"
#include "utils/color_out.h"
#include "utils/macos.h"
#include "utils/ring_buffer.h"

using namespace std::chrono;
using std::cout;
using std::stoi;

constexpr int DEFAULT_BUFLEN_MS = 50;
constexpr int CA_DIR = 1;
#define NO_DATA_STOP_SEC 2
#define MOD_NAME "[CoreAudio play.] "

struct state_ca_playback {
        AudioComponentInstance auHALComponentInstance;
        struct audio_desc desc;
        void *buffer; // audio buffer
        struct audio_buffer_api *buffer_fns;
        int audio_packet_size;
        steady_clock::time_point last_audio_read;
        bool quiet; ///< do not report buffer underruns if we do not receive data at all for a long period
        bool initialized;
        int buf_len_ms = DEFAULT_BUFLEN_MS;
};

static OSStatus theRenderProc(void *inRefCon,
                              AudioUnitRenderActionFlags *inActionFlags,
                              const AudioTimeStamp *inTimeStamp,
                              UInt32 inBusNumber, UInt32 inNumFrames,
                              AudioBufferList *ioData);

static OSStatus theRenderProc(void *inRefCon,
                              AudioUnitRenderActionFlags *inActionFlags,
                              const AudioTimeStamp *inTimeStamp,
                              UInt32 inBusNumber, UInt32 inNumFrames,
                              AudioBufferList *ioData)
{
        UNUSED(inActionFlags);
        UNUSED(inTimeStamp);
        UNUSED(inBusNumber);

        struct state_ca_playback * s = (struct state_ca_playback *) inRefCon;
        int write_bytes = inNumFrames * s->audio_packet_size;

        int ret = s->buffer_fns->read(s->buffer, (char *) ioData->mBuffers[0].mData, write_bytes);
        ioData->mBuffers[0].mDataByteSize = ret;

        if(ret < write_bytes) {
                if (!s->quiet) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Audio buffer underflow (" << write_bytes << " requested, " << ret << " written).\n";
                }
                //memset(ioData->mBuffers[0].mData, 0, write_bytes);
                ioData->mBuffers[0].mDataByteSize = ret;
                if (!s->quiet && duration_cast<seconds>(steady_clock::now() - s->last_audio_read).count() > NO_DATA_STOP_SEC) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "No data for " << NO_DATA_STOP_SEC << " seconds! Stopping.\n";
                        s->quiet = true;
                }
        } else {
                if (s->quiet) {
                        LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Starting again.\n";
                }
                s->quiet = false;
                s->last_audio_read = steady_clock::now();
        }
        return noErr;
}

static bool audio_play_ca_ctl(void *state [[gnu::unused]], int request, void *data, size_t *len)
{
        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                if (*len >= sizeof(struct audio_desc)) {
                        struct audio_desc desc;
                        memcpy(&desc, data, sizeof desc);
                        desc.codec = AC_PCM;
                        memcpy(data, &desc, sizeof desc);
                        *len = sizeof desc;
                        return true;
                } else{
                        return false;
                }
        default:
                return false;
        }
}

ADD_TO_PARAM("ca-disable-adaptive-buf", "* ca-disable-adaptive-buf\n"
                "  Core Audio - use fixed audio playback buffer instead of an adaptive one\n");
static bool audio_play_ca_reconfigure(void *state, struct audio_desc desc)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;
        AudioStreamBasicDescription stream_desc;
        UInt32 size;
        OSStatus ret = noErr;
        AURenderCallbackStruct  renderStruct;

        LOG(LOG_LEVEL_NOTICE) << MOD_NAME "Audio reinitialized to " << desc.bps * 8 << "-bit, " << desc.ch_count << " channels, " << desc.sample_rate << " Hz\n";

        if (s->initialized) {
                ret = AudioOutputUnitStop(s->auHALComponentInstance);
                if(ret) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Cannot stop AUHAL instance: " << get_osstatus_str(ret) << ".\n";
                        goto error;
                }

                ret = AudioUnitUninitialize(s->auHALComponentInstance);
                if(ret) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Cannot uninitialize AUHAL instance: " << get_osstatus_str(ret) << ".\n";
                        goto error;
                }
                s->initialized = false;
        }

        s->desc = desc;

        if (s->buffer_fns) {
                s->buffer_fns->destroy(s->buffer);
                s->buffer_fns = nullptr;
                s->buffer = nullptr;
        }

        {
                if (get_commandline_param("audio-disable-adaptive-buffer") != nullptr || get_commandline_param("ca-disable-adaptive-buf") != nullptr) {
                        if (get_commandline_param("ca-disable-adaptive-buf") != nullptr) {
                                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Param \"ca-disable-adaptive-buf\" is deprecated, use audio-disable-adaptive-bufer instead.\n";
                        }
                        int buf_len = desc.bps * desc.ch_count * (desc.sample_rate * s->buf_len_ms / 1000);
                        s->buffer = ring_buffer_init(buf_len);
                        s->buffer_fns = &ring_buffer_fns;
                } else {
                        s->buffer = audio_buffer_init(desc.sample_rate, desc.bps, desc.ch_count, s->buf_len_ms);
                        s->buffer_fns = &audio_buffer_fns;
                }
        }

        size = sizeof(stream_desc);
        ret = AudioUnitGetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                        0, &stream_desc, &size);
        if(ret) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Cannot get device format from AUHAL instance: " << get_osstatus_str(ret) << ".\n";
                goto error;
        }
        stream_desc.mSampleRate = desc.sample_rate;
        stream_desc.mFormatID = kAudioFormatLinearPCM;
        stream_desc.mChannelsPerFrame = desc.ch_count;
        stream_desc.mBitsPerChannel = desc.bps * 8;
        stream_desc.mFormatFlags = kAudioFormatFlagIsSignedInteger|kAudioFormatFlagIsPacked;
        stream_desc.mFramesPerPacket = 1;
        s->audio_packet_size = stream_desc.mBytesPerFrame = stream_desc.mBytesPerPacket = stream_desc.mFramesPerPacket * desc.ch_count * desc.bps;

        ret = AudioUnitSetProperty(s->auHALComponentInstance, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,
                        0, &stream_desc, sizeof(stream_desc));
        if(ret) {
                LOG(LOG_LEVEL_ERROR) << "Cannot set device format to AUHAL instance: " << get_osstatus_str(ret) << ".\n";
                goto error;
        }

        renderStruct.inputProc = theRenderProc;
        renderStruct.inputProcRefCon = s;
        ret = AudioUnitSetProperty(s->auHALComponentInstance, kAudioUnitProperty_SetRenderCallback,
                        kAudioUnitScope_Input, 0, &renderStruct, sizeof(AURenderCallbackStruct));
        if(ret) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Cannot register audio processing callback: " << get_osstatus_str(ret) << ".\n";
                goto error;
        }

        ret = AudioUnitInitialize(s->auHALComponentInstance);
        if(ret) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Cannot initialize AUHAL: " << get_osstatus_str(ret) << ".\n";
                goto error;
        }

        ret = AudioOutputUnitStart(s->auHALComponentInstance);
        if(ret) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Cannot start AUHAL: " << get_osstatus_str(ret) << ".\n";
                goto error;
        }

        s->initialized = true;

        return true;

error:
        return false;
}


// https://stackoverflow.com/questions/4575408/audioobjectgetpropertydata-to-get-a-list-of-input-devices#answer-4577271
static bool is_requested_direction(AudioObjectPropertyAddress propertyAddress, AudioDeviceID *audioDevice) {
        // Determine if the device is an input/output device (it is an input/output device if it has input/output channels)
        UInt32 size = 0;
        OSStatus status;
        propertyAddress.mSelector = kAudioDevicePropertyStreamConfiguration;
        status = AudioObjectGetPropertyDataSize(*audioDevice, &propertyAddress, 0, NULL, &size);
        if(kAudioHardwareNoError != status) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "AudioObjectGetPropertyDataSize (kAudioDevicePropertyStreamConfiguration) failed: " << status << "\n";
                return false;
        }

        AudioBufferList *bufferList = static_cast<AudioBufferList *>(malloc(size));
        if(NULL == bufferList) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Unable to allocate memory\n";
                return false;
        }
        status = AudioObjectGetPropertyData(*audioDevice, &propertyAddress, 0, NULL, &size, bufferList);
        if(kAudioHardwareNoError != status || 0 == bufferList->mNumberBuffers) {
                if(kAudioHardwareNoError != status)
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "AudioObjectGetPropertyData (kAudioDevicePropertyStreamConfiguration) failed: " << status << "\n";
                free(bufferList);
                return false;
        }
        free(bufferList);
        return true;
}

void audio_ca_get_device_name(AudioDeviceID dev_id, size_t namebuf_len, char *namebuf)
{
        AudioObjectPropertyAddress propertyAddress{};
        propertyAddress.mSelector = kAudioHardwarePropertyDevices;
        propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
        propertyAddress.mElement = kAudioObjectPropertyElementMain;
        propertyAddress.mSelector = kAudioDevicePropertyDeviceNameCFString;
        CFStringRef deviceName = NULL;
        UInt32 size = sizeof(deviceName);
        if (OSStatus ret = AudioObjectGetPropertyData(dev_id, &propertyAddress, 0, NULL, &size, &deviceName)) {
                log_msg(LOG_LEVEL_WARNING, "[CoreAudio] Cannot get device %" PRIu32 " name: %s\n",
                        dev_id, get_osstatus_str(ret));
                snprintf(namebuf, namebuf_len, "CoreAudio device #%" PRIu32 " (unable to get name)", dev_id);
                return;
        }
        CFStringGetCString(deviceName, namebuf, namebuf_len, kCFStringEncodingMacRoman);
        CFRelease(deviceName);
}

/**
 * @briew probe for available capture/playback devices (direction given by
 * parameter dir)
 * @param dir -1 capture, 1 playback
 */
void audio_ca_probe(struct device_info **available_devices, int *count, int dir)
{
        assert(dir == -1 || dir == 1);
        *available_devices =
            (struct device_info *) calloc(1, sizeof(struct device_info));
        assert(*available_devices != nullptr);
        snprintf((*available_devices)[0].dev, sizeof(*available_devices)[0].dev,
                 "");
        snprintf((*available_devices)[0].name,
                 sizeof(*available_devices)[0].name, "Default CoreAudio %s",
                 dir == -1 ? "capture" : "playback");
        *count = 1;

        int dev_count;
        AudioDeviceID *dev_ids;
        OSStatus ret;
        UInt32 size;
        AudioObjectPropertyAddress propertyAddress;

        propertyAddress.mSelector = kAudioHardwarePropertyDevices;
        propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
        propertyAddress.mElement = kAudioObjectPropertyElementMain;
        ret = AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &size);
        if (ret) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot get device list size: %s\n", get_osstatus_str(ret));
                goto error;
        }
        dev_ids = (AudioDeviceID *) malloc(size);
        dev_count = size / sizeof(AudioDeviceID);
        ret = AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &size, dev_ids);
        if (ret) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot get device list: %s\n", get_osstatus_str(ret));
                goto error;
        }

        propertyAddress.mScope = dir == -1 ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput;
        for (int i = 0; i < dev_count; ++i) {
                if (dir != 0 && !is_requested_direction(propertyAddress, &dev_ids[i])) {
                        continue;
                }

                (*count)++;
                *available_devices = (struct device_info *) realloc(*available_devices, *count * sizeof(struct device_info));
                assert(*available_devices != nullptr);
                memset(&(*available_devices)[*count - 1], 0, sizeof(struct device_info));

                audio_ca_get_device_name(dev_ids[i], sizeof (*available_devices)[*count - 1].name,
                                         (char *) (*available_devices)[*count - 1].name);
                snprintf((*available_devices)[*count - 1].dev, sizeof (*available_devices)[*count - 1].dev,
                                ":%" PRIu32, dev_ids[i]);
        }
        free(dev_ids);

        return;

error:
        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Error obtaining device list.\n";
}

/**
 * @briew tries to get device id from device name
 * @param name  name of the device to be looked up
 * @param dir   direction - @sa audio_ca_probe for values;
 *              dir is needed only for default device names
 */
AudioDeviceID
audio_ca_get_device_by_name(const char *name, int dir)
{
        struct device_info *devices = NULL;
        int count = 0;
        audio_ca_probe(&devices, &count, dir);
        for (int i = 0; i < count; ++i) {
                if (strstr(devices[i].name, name) != NULL) {
                        return atoi(devices[i].dev + 1);
                }
        }
        return UINT_MAX;
}

static void audio_play_ca_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        audio_ca_probe(available_devices, count, 1);
}

static void audio_play_ca_help()
{
        cout << "Core Audio playback usage:\n";
        col() << SBOLD(SRED("\t-r coreaudio") << "[:<index>|:<name>] "
                "[--param audio-buffer-len=<len_ms>] [--param audio-disable-adaptive-buffer]") << "\n";
        col() << "where\n\t" << SBOLD("<name>") << " - device name substring (case sensitive)\n\n";
        printf("Available CoreAudio playback devices:\n");
        struct device_info *available_devices;
        int count;
        void (*deleter)(void *);
        audio_play_ca_probe(&available_devices, &count, &deleter);

        for (int i = 0; i < count; ++i) {
                color_printf("\t" TBOLD("coreaudio%-4s") ": %s\n", available_devices[i].dev, available_devices[i].name);
        }
        deleter ? deleter(available_devices) : free(available_devices);
}

static void *
audio_play_ca_init(const struct audio_playback_opts *opts)
{
        if (strcmp(opts->cfg, "help") == 0) {
                audio_play_ca_help();
                return INIT_NOERR;
        }

        OSStatus ret = noErr;
        AudioComponent comp;
        AudioComponentDescription comp_desc;
        AudioDeviceID device;

        struct state_ca_playback *s = new struct state_ca_playback();

        if (const char *val = get_commandline_param("audio-buffer-len")) {
                s->buf_len_ms = atoi(val);
                if (s->buf_len_ms <= 0 || s->buf_len_ms >= 10000) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Wrong value \"" <<  val << "\" given to \"audio-buffer-len\", allowed range (0, 10000).\n";
                        goto error;
                }
        }

        //There are several different types of Audio Units.
        //Some audio units serve as Outputs, Mixers, or DSP
        //units. See AUComponent.h for listing
        comp_desc.componentType = kAudioUnitType_Output;

        //Every Component has a subType, which will give a clearer picture
        //of what this components function will be.
        //comp_desc.componentSubType = kAudioUnitSubType_DefaultOutput;
        comp_desc.componentSubType = kAudioUnitSubType_HALOutput;

        //all Audio Units in AUComponent.h must use
        //"kAudioUnitManufacturer_Apple" as the Manufacturer
        comp_desc.componentManufacturer = kAudioUnitManufacturer_Apple;
        comp_desc.componentFlags = 0;
        comp_desc.componentFlagsMask = 0;

        comp = AudioComponentFindNext(NULL, &comp_desc);
        if (!comp) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot find audio component!\n");
                goto error;
        }
        ret = AudioComponentInstanceNew(comp, &s->auHALComponentInstance);
        if (ret != noErr) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot open audio component: %s\n", get_osstatus_str(ret));
                goto error;
        }

        s->buffer = NULL;

        if ((ret = AudioUnitUninitialize(s->auHALComponentInstance)) != noErr) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot initialize audio unit: %s\n", get_osstatus_str(ret));
                goto error;
        }

        if (strlen(opts->cfg) > 0) {
                try {
                        device = stoi(opts->cfg);
                } catch (std::invalid_argument &e) {
                        device = audio_ca_get_device_by_name(opts->cfg, CA_DIR);
                        if (device == UINT_MAX) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME
                                        "Wrong device index "
                                        "or unrecognized name \"%s\"!\n",
                                        opts->cfg);
                                goto error;
                        }
                }
        } else {
                AudioObjectPropertyAddress propertyAddress;
                UInt32 size = sizeof device;
                propertyAddress.mSelector = kAudioHardwarePropertyDefaultOutputDevice;
                propertyAddress.mScope = kAudioObjectPropertyScopeGlobal;
                propertyAddress.mElement = kAudioObjectPropertyElementMain;
                if ((ret = AudioObjectGetPropertyData(kAudioObjectSystemObject, &propertyAddress, 0, NULL, &size, &device)) != noErr) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot get default audio device: %s\n", get_osstatus_str(ret));
                        goto error;
                }
        }
        char device_name[128];
        audio_ca_get_device_name(device, sizeof device_name, device_name);
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Using device: %s\n", device_name);

        if (get_commandline_param("ca-disable-adaptive-buf") == nullptr &&
                        get_commandline_param("audio-disable-adaptive-buffer") == nullptr) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Using adaptive buffer. "
                        "In case of problems, try \"--param audio-disable-adaptive-buffer\" "
                        "option.\n";
        }

        ret = AudioUnitSetProperty(s->auHALComponentInstance,
                         kAudioOutputUnitProperty_CurrentDevice,
                         kAudioUnitScope_Global,
                         1,
                         &device,
                         sizeof(device));
        if (ret) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot set audio properties: %s\n", get_osstatus_str(ret));
                goto error;
        }

        return s;

error:
        delete s;
        return NULL;
}

static void audio_play_ca_put_frame(void *state, const struct audio_frame *frame)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;

        s->buffer_fns->write(s->buffer, frame->data, frame->data_len);
}

static void audio_play_ca_done(void *state)
{
        struct state_ca_playback *s = (struct state_ca_playback *)state;

        if (s->initialized) {
                AudioOutputUnitStop(s->auHALComponentInstance);
                AudioUnitUninitialize(s->auHALComponentInstance);
        }
        if (s->buffer_fns) {
                s->buffer_fns->destroy(s->buffer);
        }
        delete s;
}

static const struct audio_playback_info aplay_coreaudio_info = {
        audio_play_ca_probe,
        audio_play_ca_init,
        audio_play_ca_put_frame,
        audio_play_ca_ctl,
        audio_play_ca_reconfigure,
        audio_play_ca_done
};

REGISTER_MODULE(coreaudio, &aplay_coreaudio_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

