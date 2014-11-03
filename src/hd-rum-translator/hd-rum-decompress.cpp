#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "hd-rum-translator/hd-rum-decompress.h"
#include "hd-rum-translator/hd-rum-recompress.h"

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "debug.h"
#include "host.h"
#include "rtp/rtp.h"

#include "video.h"
#include "video_display.h"
#include "video_rxtx/ultragrid_rtp.h"

static constexpr int MAX_QUEUE_SIZE = 2;

using namespace std;

namespace hd_rum_decompress {
struct state_transcoder_decompress : public frame_recv_delegate {
        struct output_port_info {
                inline output_port_info(void *s, bool a) : state(s), active(a) {}
                void *state;
                bool active;
        };

        struct message {
                inline message(shared_ptr<video_frame> && f) : type(FRAME), frame(std::move(f)) {}
                inline message() : type(QUIT) {}
                inline message(int ri) : type(REMOVE_INDEX), remove_index(ri) {}
                inline message(void *ns) : type(NEW_RECOMPRESS), new_recompress_state(ns) {}
                inline message(message && original);
                inline ~message();
                enum { FRAME, REMOVE_INDEX, NEW_RECOMPRESS, QUIT } type;
                union {
                        shared_ptr<video_frame> frame;
                        int remove_index;
                        void *new_recompress_state;
                };
        };

        vector<output_port_info> output_ports;

        unique_ptr<ultragrid_rtp_video_rxtx> video_rxtx;

        queue<message> received_frame;

        mutex              lock;
        condition_variable have_frame_cv;
        condition_variable frame_consumed_cv;
        thread             worker_thread;

        struct display *display;
        struct control_state *control;
        thread         receiver_thread;

        void frame_arrived(struct video_frame *f);

        virtual ~state_transcoder_decompress() {}
        void worker();

        thread             display_thread;
};

void state_transcoder_decompress::frame_arrived(struct video_frame *f)
{
        unique_lock<mutex> l(lock);
        if (received_frame.size() >= MAX_QUEUE_SIZE) {
                fprintf(stderr, "Hd-rum-decompress max queue size (%d) reached!\n", MAX_QUEUE_SIZE);
        }
        frame_consumed_cv.wait(l, [this]{ return received_frame.size() < MAX_QUEUE_SIZE; });
        received_frame.push(shared_ptr<video_frame>(f, vf_free));
        l.unlock();
        have_frame_cv.notify_one();
}

inline state_transcoder_decompress::message::message(message && original)
        : type(original.type)
{
        switch (original.type) {
                case FRAME:
                        new (&frame) shared_ptr<video_frame>(std::move(original.frame));
                        break;
                case REMOVE_INDEX:
                        remove_index = original.remove_index;
                        break;
                case NEW_RECOMPRESS:
                        new_recompress_state = original.new_recompress_state;
                        break;
                case QUIT:
                        break;
        }
}

inline state_transcoder_decompress::message::~message() {
        // shared_ptr has non-trivial destructor
        if (type == FRAME) {
                frame.~shared_ptr<video_frame>();
        }
}
} // end of hd-rum-decompress namespace

using namespace hd_rum_decompress;

void hd_rum_decompress_set_active(void *state, void *recompress_port, bool active)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        for (auto && port : s->output_ports) {
                if (port.state == recompress_port) {
                        port.active = active;
                }
        }
}

ssize_t hd_rum_decompress_write(void *state, void *buf, size_t count)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        return rtp_send_raw_rtp_data(s->video_rxtx->m_network_devices[0],
                        (char *) buf, count);
}

void state_transcoder_decompress::worker()
{
        bool should_exit = false;
        while (!should_exit) {
                unique_lock<mutex> l(lock);
                have_frame_cv.wait(l, [this]{return !received_frame.empty();});

                message msg(std::move(received_frame.front()));
                l.unlock();

                switch (msg.type) {
                case message::QUIT:
                        should_exit = true;
                        break;
                case message::REMOVE_INDEX:
                        recompress_done(output_ports[msg.remove_index].state);
                        output_ports.erase(output_ports.begin() + msg.remove_index);
                        break;
                case message::NEW_RECOMPRESS:
                        output_ports.emplace_back(msg.new_recompress_state, true);
                        break;
                case message::FRAME:
                        for (unsigned int i = 0; i < output_ports.size(); ++i) {
                                if (output_ports[i].active)
                                        recompress_process_async(output_ports[i].state, msg.frame);
                        }
                        break;
                }

                // we are removing from queue now because special messages are "accepted" when queue is empty
                l.lock();
                received_frame.pop();
                l.unlock();
                frame_consumed_cv.notify_one();
        }
}

void *hd_rum_decompress_init(struct module *parent)
{
        struct state_transcoder_decompress *s;
        bool use_ipv6 = false;

        s = new state_transcoder_decompress();

        char cfg[128] = "";
        snprintf(cfg, sizeof cfg, "pipe:%p", s);
        assert (initialize_video_display(parent, "proxy", cfg, 0, &s->display) == 0);

        map<string, param_u> params;

        // common
        params["parent"].ptr = parent;
        params["exporter"].ptr = NULL;
        params["compression"].ptr = (void *) "none";
        params["rxtx_mode"].i = MODE_RECEIVER;

        //RTP
        params["mtu"].i = 9000; // doesn't matter anyway...
        // should be localhost and RX TX ports the same (here dynamic) in order to work like a pipe
        params["receiver"].ptr = (void *) "localhost";
        params["rx_port"].i = 0;
        params["tx_port"].i = 0;
        params["use_ipv6"].b = use_ipv6;
        params["mcast_if"].ptr = (void *) NULL;
        params["fec"].ptr = (void *) "none";
        params["encryption"].ptr = (void *) NULL;
        params["packet_rate"].i = 0;

        // UltraGrid RTP
        params["postprocess"].ptr = (void *) NULL;
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = s->display;

        s->video_rxtx = unique_ptr<ultragrid_rtp_video_rxtx>(dynamic_cast<ultragrid_rtp_video_rxtx *>(video_rxtx::create(ULTRAGRID_RTP, params)));
        assert (s->video_rxtx);

        s->worker_thread = thread(&state_transcoder_decompress::worker, s);
        s->receiver_thread = thread(&video_rxtx::receiver_thread, s->video_rxtx.get());
        s->display_thread = thread(display_run, s->display);

        return (void *) s;
}

void hd_rum_decompress_done(void *state) {
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        should_exit_receiver = true;
        s->receiver_thread.join();

        {
                unique_lock<mutex> l(s->lock);
                s->received_frame.push({});
                l.unlock();
                s->have_frame_cv.notify_one();
        }

        s->worker_thread.join();

        // cleanup
        for (unsigned int i = 0; i < s->output_ports.size(); ++i) {
                recompress_done(s->output_ports[i].state);
        }

        display_put_frame(s->display, NULL, 0);
        s->display_thread.join();
        display_done(s->display);

        delete s;
}

void hd_rum_decompress_remove_port(void *state, int index) {
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        unique_lock<mutex> l(s->lock);
        s->received_frame.push(index);
        s->have_frame_cv.notify_one();
        s->frame_consumed_cv.wait(l, [s]{ return s->received_frame.size() == 0; });
}


void hd_rum_decompress_append_port(void *state, void *recompress_state)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        unique_lock<mutex> l(s->lock);
        s->received_frame.push(recompress_state);
        s->have_frame_cv.notify_one();
        s->frame_consumed_cv.wait(l, [s]{ return s->received_frame.size() == 0; });
}

