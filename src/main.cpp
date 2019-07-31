/*
 * FILE:    main.cpp
 * AUTHORS: Colin Perkins    <csp@csperkins.org>
 *          Ladan Gharai     <ladan@isi.edu>
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *          David Cassany    <david.cassany@i2cat.net>
 *          Ignacio Contreras <ignacio.contreras@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec     <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2005-2018 CESNET z.s.p.o.
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2003-2004 University of Glasgow
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
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
 *
 */

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <pthread.h>
#include <rang.hpp>

#include "control_socket.h"
#include "debug.h"
#include "host.h"
#include "keyboard_control.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "rtp/rtp.h"
#include "rtsp/rtsp_utils.h"
#include "ug_runtime_error.h"
#include "utils/misc.h"
#include "utils/net.h"
#include "utils/thread.h"
#include "utils/wait_obj.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture/import.h"
#include "video_display.h"
#include "video_compress.h"
#include "export.h"
#include "video_rxtx.h"
#include "audio/audio.h"
#include "audio/audio_capture.h"
#include "audio/codec.h"
#include "audio/utils.h"

#include <iostream>
#include <memory>
#include <string>

#define PORT_BASE               5004

#define DEFAULT_AUDIO_FEC       "none"
static constexpr const char *DEFAULT_VIDEO_COMPRESSION = "none";
static constexpr const char *DEFAULT_AUDIO_CODEC = "PCM";

#define OPT_AUDIO_CAPTURE_CHANNELS (('a' << 8) | 'c')
#define OPT_AUDIO_CAPTURE_FORMAT (('C' << 8) | 'F')
#define OPT_AUDIO_CHANNEL_MAP (('a' << 8) | 'm')
#define OPT_AUDIO_CODEC (('A' << 8) | 'C')
#define OPT_AUDIO_DELAY (('A' << 8) | 'D')
#define OPT_AUDIO_PROTOCOL (('A' << 8) | 'P')
#define OPT_AUDIO_SCALE (('a' << 8) | 's')
#define OPT_CAPABILITIES (('C' << 8) | 'C')
#define OPT_CAPTURE_FILTER (('O' << 8) | 'F')
#define OPT_CONTROL_PORT (('C' << 8) | 'P')
#define OPT_CUDA_DEVICE (('C' << 8) | 'D')
#define OPT_ECHO_CANCELLATION (('E' << 8) | 'C')
#define OPT_ENCRYPTION (('E' << 8) | 'N')
#define OPT_FULLHELP (('F' << 8u) | 'H')
#define OPT_EXPORT (('E' << 8) | 'X')
#define OPT_IMPORT (('I' << 8) | 'M')
#define OPT_LIST_MODULES (('L' << 8) | 'M')
#define OPT_MCAST_IF (('M' << 8) | 'I')
#define OPT_PARAM (('O' << 8) | 'P')
#define OPT_PIX_FMTS (('P' << 8) | 'F')
#define OPT_PROTOCOL (('P' << 8) | 'R')
#define OPT_START_PAUSED (('S' << 8) | 'P')
#define OPT_VERBOSE (('V' << 8) | 'E')
#define OPT_VIDEO_CODECS (('V' << 8) | 'C')
#define OPT_VIDEO_PROTOCOL (('V' << 8) | 'P')
#define OPT_WINDOW_TITLE (('W' << 8) | 'T')

#define MAX_CAPTURE_COUNT 17

using rang::fg;
using rang::style;
using namespace std;

struct state_uv {
        state_uv() : capture_device{}, display_device{}, audio{}, state_video_rxtx{} {
                module_init_default(&root_module);
                root_module.cls = MODULE_CLASS_ROOT;
                root_module.priv_data = this;
        }
        ~state_uv() {
                module_done(&root_module);
        }

        struct vidcap *capture_device;
        struct display *display_device;

        struct state_audio *audio;

        struct module root_module;

        video_rxtx *state_video_rxtx;
};

static int exit_status = EXIT_SUCCESS;

static struct state_uv *uv_state;

static void signal_handler(int signal)
{
        if (log_level >= LOG_LEVEL_DEBUG) {
                char msg[] = "Caught signal ";
                char buf[128];
                char *ptr = buf;
                for (size_t i = 0; i < sizeof msg - 1; ++i) {
                        *ptr++ = msg[i];
                }
                if (signal / 10) {
                        *ptr++ = '0' + signal/10;
                }
                *ptr++ = '0' + signal%10;
                *ptr++ = '\n';
                size_t bytes = ptr - buf;
                ptr = buf;
                do {
                        ssize_t written = write(STDERR_FILENO, ptr, bytes);
                        if (written < 0) {
                                break;
                        }
                        bytes -= written;
                        ptr += written;
                } while (bytes > 0);
        }
        exit_uv(0);
}

static void crash_signal_handler(int sig)
{
        char buf[1024];
        char *ptr = buf;
        *ptr++ = '\n';
        const char message1[] = " has crashed";
        for (size_t i = 0; i < sizeof PACKAGE_NAME - 1; ++i) {
                *ptr++ = PACKAGE_NAME[i];
        }
        for (size_t i = 0; i < sizeof message1 - 1; ++i) {
                *ptr++ = message1[i];
        }
#ifndef WIN32
        *ptr++ = ' '; *ptr++ = '(';
        for (size_t i = 0; i < sizeof sys_siglist[sig] - 1; ++i) {
                if (sys_siglist[sig][i] == '\0') {
                        break;
                }
                *ptr++ = sys_siglist[sig][i];
        }
        *ptr++ = ')';
#endif
        const char message2[] = ".\n\nPlease send a bug report to address ";
        for (size_t i = 0; i < sizeof message2 - 1; ++i) {
                *ptr++ = message2[i];
        }
        for (size_t i = 0; i < sizeof PACKAGE_BUGREPORT - 1; ++i) {
                *ptr++ = PACKAGE_BUGREPORT[i];
        }
        *ptr++ = '.'; *ptr++ = '\n';
        const char message3[] = "You may find some tips how to report bugs in file REPORTING-BUGS distributed with ";
        for (size_t i = 0; i < sizeof message3 - 1; ++i) {
                *ptr++ = message3[i];
        }
        for (size_t i = 0; i < sizeof PACKAGE_NAME - 1; ++i) {
                *ptr++ = PACKAGE_NAME[i];
        }
        *ptr++ = '.'; *ptr++ = '\n';

        size_t bytes = ptr - buf;
        ptr = buf;
        do {
                ssize_t written = write(STDERR_FILENO, ptr, bytes);
                if (written < 0) {
                        break;
                }
                bytes -= written;
                ptr += written;
        } while (bytes > 0);

        signal(SIGABRT, SIG_DFL);
        signal(SIGSEGV, SIG_DFL);
        raise(sig);
}

void exit_uv(int status) {
        exit_status = status;
        should_exit = true;
}

static void print_help_item(const string &name, const vector<string> &help) {
        int help_lines = 0;

        cout << style::bold << "\t" << name << style::reset;

        for (auto line : help) {
                int spaces = help_lines == 0 ? 31 - (int) name.length() : 39;
                for (int i = 0; i < max(spaces, 0) + 1; ++i) {
                        cout << " ";
                }
                cout << line << "\n";
                help_lines += 1;
        }

        if (help_lines == 0) {
                cout << "\n";
        }
        cout << "\n";
}

static void usage(const char *exec_path, bool full = false)
{
        cout << "Usage: " << fg::red << style::bold << (exec_path ? exec_path : "<executable_path>") << fg::reset << " [options] address\n\n" << style::reset;
        printf("Options:\n");
        print_help_item("-h | --fullhelp", {"show usage (basic/full)"});
        print_help_item("-d <display_device>", {"select display device, use '-d help'",
                        "to get list of supported devices"});
        print_help_item("-t <capture_device>", {"select capture device, use '-t help'",
                        "to get list of supported devices"});
        print_help_item("-c <cfg>", {"video compression (see '-c help')"});
        print_help_item("-r <playback_device>", {"audio playback device (see '-r help')"});
        print_help_item("-s <capture_device>", {"audio capture device (see '-s help')"});
        if (full) {
                print_help_item("--verbose[=<level>]", {"print verbose messages (optinaly specify level [0-" + to_string(LOG_LEVEL_MAX) + "])"});
                print_help_item("--list-modules", {"prints list of modules"});
                print_help_item("--control-port <port>[:0|1]", {"set control port (default port: " + to_string(DEFAULT_CONTROL_PORT) + ")",
                                "connection types: 0- Server (default), 1- Client"});
                print_help_item("--video-protocol <proto>", {"transmission protocol, see '--video-protocol help'",
                                "for list. Use --video-protocol rtsp for RTSP server",
                                "(see --video-protocol rtsp:help for usage)"});
                print_help_item("--audio-protocol <proto>[:<settings>]", {"<proto> can be " AUDIO_PROTOCOLS});
                print_help_item("--protocol <proto>", {"shortcut for '--audio-protocol <proto> --video-protocol <proto>'"});
#ifdef HAVE_IPv6
                print_help_item("-4/-6", {"force IPv4/IPv6 resolving"});
#endif //  HAVE_IPv6
                print_help_item("--mcast-if <iface>", {"bind to specified interface for multicast"});
                print_help_item("-M <video_mode>", {"received video mode (eg tiled-4K, 3D,",
                                "dual-link)"});
                print_help_item("-p <postprocess> | help", {"postprocess module"});
        }
        print_help_item("-f [A:|V:]<settings>", {"FEC settings (audio or video) - use",
                        "\"none\", \"mult:<nr>\",", "\"ldgm:<max_expected_loss>%%\" or", "\"ldgm:<k>:<m>:<c>\"",
                        "\"rs:<k>:<n>\""});
        print_help_item("-P <port> | <video_rx>:<video_tx>[:<audio_rx>:<audio_tx>]", { "",
                        "<port> is base port number, also 3",
                        "subsequent ports can be used for RTCP",
                        "and audio streams. Default: " + to_string(PORT_BASE) + ".",
                        "You can also specify all two or four", "ports directly."});
        print_help_item("-l <limit_bitrate> | unlimited | auto", {"limit sending bitrate",
                        "to <limit_bitrate> (with optional k/M/G suffix)"});
        if (full) {
                print_help_item("-A <address>", {"audio destination address",
                                "If not specified, will use same as for video"});
        }
        print_help_item("--audio-capture-format <fmt> | help", {"format of captured audio"});
        if (full) {
                print_help_item("--audio-channel-map <mapping> | help", {});
        }
        print_help_item("--audio-codec <codec>[:sample_rate=<sr>][:bitrate=<br>] | help", {"audio codec"});
        if (full) {
                print_help_item("--audio-delay <delay_ms>", {"amount of time audio should be delayed to video",
                                "(may be also negative to delay video)"});
                print_help_item("--audio-scale <factor> | <method> | help",
                                {"scales received audio"});
        }
#if 0
        printf("\t--echo-cancellation      \tapply acoustic echo cancellation to audio\n");
        printf("\n");
#endif
        print_help_item("--cuda-device <index> | help", {"use specified CUDA device"});
        if (full) {
                print_help_item("--encryption <passphrase>", {"key material for encryption"});
                print_help_item("--playback <directory> | help", {"replays recorded audio and video"});
                print_help_item("--record[=<directory>]", {"record captured audio and video"});
                print_help_item("--capture-filter <filter> | help",
                                {"capture filter(s), must be given before capture device"});
                print_help_item("--param <params> | help", {"additional advanced parameters, use help for list"});
                print_help_item("--pix-fmts", {"list of pixel formats"});
                print_help_item("--video-codecs", {"list of video codecs"});
        }
        print_help_item("address", {"destination address"});
        printf("\n");
}

/**
 * This function captures video and possibly compresses it.
 * It then delegates sending to another thread.
 *
 * @param[in] arg pointer to UltraGrid (root) module
 */
static void *capture_thread(void *arg)
{
        struct module *uv_mod = (struct module *)arg;
        struct state_uv *uv = (struct state_uv *) uv_mod->priv_data;
        struct wait_obj *wait_obj;

        wait_obj = wait_obj_init();

        while (!should_exit) {
                /* Capture and transmit video... */
                struct audio_frame *audio;
                struct video_frame *tx_frame = vidcap_grab(uv->capture_device, &audio);
                if (tx_frame != NULL) {
                        if(audio) {
                                audio_sdi_send(uv->audio, audio);
                        }
                        //tx_frame = vf_get_copy(tx_frame);
                        bool wait_for_cur_uncompressed_frame;
                        shared_ptr<video_frame> frame;
                        if (!tx_frame->callbacks.dispose) {
                                wait_obj_reset(wait_obj);
                                wait_for_cur_uncompressed_frame = true;
                                frame = shared_ptr<video_frame>(tx_frame, [wait_obj](struct video_frame *) {
                                                        wait_obj_notify(wait_obj);
                                                });
                        } else {
                                wait_for_cur_uncompressed_frame = false;
                                frame = shared_ptr<video_frame>(tx_frame, tx_frame->callbacks.dispose);
                        }

                        uv->state_video_rxtx->send(move(frame)); // std::move really important here (!)

                        // wait for frame frame to be processed, eg. by compress
                        // or sender (uncompressed video). Grab invalidates previous frame
                        // (if not defined dispose function).
                        if (wait_for_cur_uncompressed_frame) {
                                wait_obj_wait(wait_obj);
                                tx_frame->callbacks.dispose = NULL;
                                tx_frame->callbacks.dispose_udata = NULL;
                        }
                }
        }

        wait_obj_done(wait_obj);

        return NULL;
}

static bool parse_audio_capture_format(const char *optarg)
{
        if (strcmp(optarg, "help") == 0) {
                printf("Usage:\n");
                printf("\t--audio-capture-format {channels=<num>|bps=<bits_per_sample>|sample_rate=<rate>}*\n");
                printf("\t\tmultiple options can be separated by a colon\n");
                return false;
        }

        unique_ptr<char[]> arg_copy(new char[strlen(optarg) + 1]);
        char *arg = arg_copy.get();
        strcpy(arg, optarg);

        char *item, *save_ptr, *tmp;
        tmp = arg;
        char *endptr;

        while ((item = strtok_r(tmp, ":", &save_ptr))) {
                if (strncmp(item, "channels=", strlen("channels=")) == 0) {
                        item += strlen("channels=");
                        audio_capture_channels = strtol(item, &endptr, 10);
                        if (audio_capture_channels < 1 || audio_capture_channels > MAX_AUDIO_CAPTURE_CHANNELS || endptr != item + strlen(item)) {
                                log_msg(LOG_LEVEL_ERROR, "Invalid number of channels %s!\n", item);
                                return false;
                        }
                } else if (strncmp(item, "bps=", strlen("bps=")) == 0) {
                        item += strlen("bps=");
                        int bps = strtol(item, &endptr, 10);
                        if (bps % 8 != 0 || (bps != 8 && bps != 16 && bps != 24 && bps != 32) || endptr != item + strlen(item)) {
                                log_msg(LOG_LEVEL_ERROR, "Invalid bps %s!\n", item);
                                log_msg(LOG_LEVEL_ERROR, "Supported values are 8, 16, 24, or 32 bits.\n");
                                return false;

                        }
                        audio_capture_bps = bps / 8;
                } else if (strncmp(item, "sample_rate=", strlen("sample_rate=")) == 0) {
                        long long val = unit_evaluate(item + strlen("sample_rate="));
                        assert(val > 0 && val <= numeric_limits<decltype(audio_capture_sample_rate)>::max());
                        audio_capture_sample_rate = val;
                } else {
                        log_msg(LOG_LEVEL_ERROR, "Unkonwn format for --audio-capture-format!\n");
                        return false;
                }

                tmp = NULL;
        }

        return true;
}

static bool parse_params(char *optarg)
{
        if (optarg && strcmp(optarg, "help") == 0) {
                puts("Params can be one or more (separated by comma) of following:");
                print_param_doc();
                return false;
        }
        char *item, *save_ptr;
        while ((item = strtok_r(optarg, ",", &save_ptr))) {
                char *key_cstr = item;
                if (strchr(item, '=')) {
                        char *val_cstr = strchr(item, '=') + 1;
                        *strchr(item, '=') = '\0';
                        commandline_params[key_cstr] = val_cstr;
                } else {
                        commandline_params[key_cstr] = string();
                }
                if (!validate_param(key_cstr)) {
                        log_msg(LOG_LEVEL_ERROR, "Unknown parameter: %s\n", key_cstr);
                        log_msg(LOG_LEVEL_INFO, "Type '%s --param help' for list.\n", uv_argv[0]);
                        return false;
                }
                optarg = NULL;
        }
        return true;
}

#define S(A) pair<string, decoder_t>{string(#A), A}

int main(int argc, char *argv[])
{
	vector<pair<string, decoder_t>> decoders = {
		S(vc_copylineDVS10),
		S(vc_copylinev210),
		S(vc_copylineYUYV),
		S(vc_copyliner10k),
		S(vc_copylineR12L),
		S(vc_copylineRGBA),
		S(vc_copylineDVS10toV210),
		S(vc_copylineRGBAtoRGB),
		S(vc_copylineABGRtoRGB),
		S(vc_copylineRGBAtoRGBwithShift),
		S(vc_copylineRGBtoRGBA),
		S(vc_copylineRGBtoUYVY),
		S(vc_copylineRGBtoUYVY_SSE),
		S(vc_copylineRGBtoGrayscale_SSE),
		S(vc_copylineRGBtoR12L),
		S(vc_copylineR12LtoRG48),
		S(vc_copylineR12LtoRGB),
		S(vc_copylineRG48toR12L),
		S(vc_copylineRG48toRGBA),
		S(vc_copylineUYVYtoRGB),
		S(vc_copylineUYVYtoRGB_SSE),
		S(vc_copylineUYVYtoGrayscale),
		S(vc_copylineYUYVtoRGB),
		S(vc_copylineBGRtoUYVY),
		S(vc_copylineRGBAtoUYVY),
		S(vc_copylineBGRtoRGB),
		S(vc_copylineDPX10toRGBA),
		S(vc_copylineDPX10toRGB),
		S(vc_copylineRGB)
	};

	int size = 1000*1000*1000;
	unsigned char *in = (unsigned char *) calloc(1, size);
	unsigned char *out = (unsigned char *) malloc(size);
	int data_len = (size / 6) / 48 * 48;
	for (int n = 0; n < data_len; ++n) {
		out[n] = n % 255;
	}

	for (auto a: decoders) {
		auto t0 = std::chrono::high_resolution_clock::now();
		a.second(out, in, data_len, 0, 8, 16);
		cout << a.first << ": " << chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now() - t0).count() << "\n";
	}
}

/* vim: set expandtab sw=8: */
