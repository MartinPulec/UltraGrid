// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#ifndef VIDEO_RECV_H_EB8ECD93_2CFE_4A8A_8BBA_7D9C38EBC6CE
#define VIDEO_RECV_H_EB8ECD93_2CFE_4A8A_8BBA_7D9C38EBC6CE

struct display;
struct module;
struct rxtx;
struct rxtx_params;

struct state_video_recv *video_recv_start(struct rxtx              *rxtx,
                                          const struct rxtx_params *params,
                                          struct module            *parent,
                                          struct display           *d);
void                     video_recv_join(struct state_video_recv *s);
void                     video_recv_done(struct state_video_recv *s);

#endif // !defined VIDEO_RECV_H_EB8ECD93_2CFE_4A8A_8BBA_7D9C38EBC6CE
