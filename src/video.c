// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#include "video.h"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include "compat/c23.h"   // IWYU pragma: keep
#include "module.h"       // for module, module_class, module_init_default
#include "utils/macros.h" // for to_fourcc
#include "video_recv.h"   // for video_recv_done, video_recv_join, video_re...

#define MOD_NAME "[video] "
#define MAGIC to_fourcc('v', 'd', 'e', 'o')

struct state_video {
        uint32_t                 magic;
        struct module            mod;
        struct state_video_recv *recv_state;
};

struct state_video *
video_start(struct rxtx *rxtx, const struct rxtx_params *params,
            struct module *parent, struct display *d)
{
        struct state_video *s = calloc(1, sizeof *s);
        s->magic              = MAGIC;

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_VIDEO;
        module_register(&s->mod, parent);

        s->recv_state = video_recv_start(rxtx, params, &s->mod, d);
        if (s->recv_state == nullptr) {
                video_done(s);
                return nullptr;
        }

        return s;
}

void
video_join(struct state_video *s)
{
        if (s == nullptr) {
                return;
        }
        video_recv_join(s->recv_state);
}

void
video_done(struct state_video *s)
{
        if (!s) {
                return;
        }
        assert(s->magic == MAGIC);
        video_recv_done(s->recv_state);
        module_done(&s->mod);
        free(s);
}
