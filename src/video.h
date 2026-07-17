// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#ifndef VIDEO_H_17645EA3_F3A7_4D11_9F77_18A41B5A6FAB
#define VIDEO_H_17645EA3_F3A7_4D11_9F77_18A41B5A6FAB

struct display;
struct module;
struct rxtx;
struct rxtx_params;

#ifdef __cplusplus
extern "C" {
#endif

struct state_video *video_start(struct rxtx              *rxtx,
                                const struct rxtx_params *params,
                                struct module *parent, struct display *d);
void                video_join(struct state_video *s);
void                video_done(struct state_video *s);

#ifdef __cplusplus
}
#endif

#endif // !defined VIDEO_H_17645EA3_F3A7_4D11_9F77_18A41B5A6FAB
