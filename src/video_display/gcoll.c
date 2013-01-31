/*
 * FILE:   video_display/gcoll.c
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *
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
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "host.h"

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>

#include <tv.h>

#ifdef HAVE_MACOSX
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h> // CGL
#include <OpenGL/glext.h>
#include <GLUT/glut.h>
#elif defined HAVE_LINUX
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glut.h>
#include "x11_common.h"
#else // WIN32
#include <GL/glew.h>
#include <GL/glut.h>
#include "compat/inet_ntop.h"
#endif /* HAVE_MACOSX */

#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif /* FREEGLUT */

#include "debug.h"
#include "video_display.h" // DISPLAY_PROPERTY_* macros
#include "video_display/gcoll.h"

#define MAGIC_GCOLL     DISPLAY_GCOLL_ID
// Length of participant timeout in seconds
#define PARTICIPANT_TIMEOUT 5
#define GROUP_TIMEOUT PARTICIPANT_TIMEOUT
#define DEFAULT_RAP_PORT 9876
#define RAP_REQUEST_LENGTH_LIMIT 1024
#define RAP_REPLY_LENGTH_LIMIT 1024
#define RAP_MAGIC_CODE_START 1000
#define RAP_MAGIC_CODE_REDIRECT (RAP_MAGIC_CODE_START + 1)
#define UNDEFINED_FD -1
#define MAX_CLIENTS 20
#define MAX_GROUPS 4
#define MAX_ADDRESS_LENGTH 45
#define SMALL_IMAGE_BORDERS 5
#define WIN_NAME "GColl via Ultragrid"

/* defined in main.c */
extern int uv_argc;
extern char **uv_argv;

struct state_gcoll;

static void setup_texture_parameters(void);

struct participant {
  uint32_t ssrc;
  uint32_t side_ssrc;
  uint32_t gaze_ssrc;

  unsigned int room;

  struct video_frame *frame;
  struct video_frame *side_frame;
  GLuint texture;
  GLuint side_texture;

  time_t ts;
};

struct group {
  unsigned int id;

  uint32_t ssrc;

  struct video_frame *frame;
  GLuint texture;

  time_t ts;
};

struct frame_storage {
  uint32_t ssrc[2 * MAX_CLIENTS + MAX_GROUPS];
  struct video_frame *frames[2 * MAX_CLIENTS + MAX_GROUPS];
  int count;
};

struct state_gcoll {
  struct gcoll_init_params *params;

  uint32_t gaze_ssrc;
  bool gaze_changed;

  struct rum_communicator *rum;
  pthread_t rum_thread;

  struct participant *participants;
  struct group *groups;

  int participants_count;
  int groups_count;

  bool participants_modified;
  bool groups_modified;

  pthread_mutex_t participants_lock;
  pthread_mutex_t groups_lock;
  pthread_mutex_t new_frames_lock;

  struct frame_storage *current_frames;
  struct frame_storage *new_frames;

  int glut_window;
  int small_images_count;
  bool received_frame;

  bool exit;

  uint32_t magic;

  struct timeval tv;
  unsigned long int calls;
};

/** Static variable for GLUT callbacks which do not accept parameters. */
static struct state_gcoll *gcoll;

static void participant_init(struct participant *p) {
  assert(p != NULL);

  p->ssrc = p->side_ssrc = p->gaze_ssrc = 0;
  p->room = 0;
  p->frame = NULL;
  p->side_frame = NULL;
  p->ts = 0;
}

static inline void participant_update_ts(struct participant *p) {
  assert(p != NULL);

  p->ts = time(NULL);
}

static void group_init(struct group *g) {
  assert(g != NULL);

  g->id = 0;
  g->ssrc = 0;
  g->frame = NULL;
  g->ts = 0;
}

static inline void group_update_ts(struct group *g) {
  assert(g != NULL);

  g->ts = time(NULL);
}

static void display_gcoll_remove_participant(struct state_gcoll *s, int index) {
  assert(s != NULL &&
      s->participants != NULL &&
      index >= 0 &&
      index < s->participants_count);

  glDeleteTextures(1, &s->participants[index].texture);
  s->participants[index].texture = 0;
  if (index != s->participants_count - 1)
    memcpy(&s->participants[index],
        &s->participants[index+1],
        sizeof(struct participant) * (s->participants_count - index - 1));
  s->participants_count--;
  s->participants_modified = true;
}

static void add_or_update_participant(struct state_gcoll *s, uint32_t ssrc, uint32_t side_ssrc, uint32_t gaze_ssrc, unsigned int room) {
  assert(s != NULL &&
      s->participants != NULL);

  bool found = false;

  pthread_mutex_lock(&s->participants_lock);
  for (int i = 0; i < s->participants_count; i++) {
    if (s->participants[i].ssrc == ssrc) {
      s->participants[i].side_ssrc = side_ssrc;
      s->participants[i].gaze_ssrc = gaze_ssrc;
      found = true;
      break;
    }
  }
  if (!found && s->participants_count < MAX_CLIENTS) {
    s->participants[s->participants_count].ssrc = ssrc;
    s->participants[s->participants_count].side_ssrc = side_ssrc;
    s->participants[s->participants_count].gaze_ssrc = gaze_ssrc;
    s->participants[s->participants_count].room = room;
    s->participants[s->participants_count].frame = NULL;
    s->participants[s->participants_count].side_frame = NULL;
    s->participants[s->participants_count].texture = 0;
    s->participants[s->participants_count].side_texture = 0;
    //glGenTextures(1, &s->participants[s->participants_count].texture);
    //glBindTexture(GL_TEXTURE_2D, s->participants[s->participants_count].texture);
    //setup_texture_parameters();
    participant_update_ts(&s->participants[s->participants_count]);
    s->participants_count++;
    s->participants_modified = true;
  }
  pthread_mutex_unlock(&s->participants_lock);
}

static void remove_inactive_participants(struct state_gcoll *s) {
  assert(s != NULL &&
      s->participants != NULL);

  time_t now = time(NULL);
  pthread_mutex_lock(&s->participants_lock);
  for (int i = s->participants_count - 1; i >= 0; i--) {
    if (s->participants[i].ts + PARTICIPANT_TIMEOUT < now)
      display_gcoll_remove_participant(s, i);
  }
  pthread_mutex_unlock(&s->participants_lock);
}

/**
 * IMPORTANT: Lock participants_lock before calling
 */
static int find_participant(struct state_gcoll *s, uint32_t ssrc) {
  assert(s != NULL &&
      s->participants != NULL);

  for (int i = 0; i < s->participants_count; i++) {
    if (s->participants[i].ssrc == ssrc)
      return i;
  }

  return -1;
}

static int find_participant_side(struct state_gcoll *s, uint32_t ssrc) {
  assert(s != NULL &&
      s->participants != NULL);

  for (int i = 0; i < s->participants_count; i++) {
    if (s->participants[i].side_ssrc == ssrc)
      return i;
  }

  return -1;
}

static void display_gcoll_remove_group(struct state_gcoll *s, int index) {
  assert(s != NULL &&
      s->groups != NULL &&
      index >= 0 &&
      index < s->groups_count);

  glDeleteTextures(1, &s->groups[index].texture);
  s->groups[index].texture = 0;
  if (index != s->groups_count - 1)
    memcpy(&s->groups[index],
        &s->groups[index + 1],
        sizeof(struct group) * (s->groups_count - index - 1));
  s->groups_count--;
  s->groups_modified = true;
}

static void add_or_update_group(struct state_gcoll *s, uint32_t ssrc, unsigned int room) {
  assert(s != NULL &&
      s->groups != NULL);

  bool found = false;

  pthread_mutex_lock(&s->groups_lock);
  for (int i = 0; i < s->groups_count; i++) {
    if (s->groups[i].ssrc == ssrc) {
      found = true;
      break;
    }
  }
  if (!found && s->groups_count < MAX_GROUPS) {
    s->groups[s->groups_count].ssrc = ssrc;
    s->groups[s->groups_count].id = room;
    s->groups[s->groups_count].frame = NULL;
    s->groups[s->groups_count].texture = 0;
    /*
       glGenTextures(1, &s->groups[s->groups_count].texture);
       glBindTexture(GL_TEXTURE_2D, s->groups[s->groups_count].texture);
       setup_texture_parameters();
       */
    group_update_ts(&s->groups[s->groups_count]);
    s->groups_count++;
    s->groups_modified = true;
  }
  pthread_mutex_unlock(&s->groups_lock);
}

static void remove_inactive_groups(struct state_gcoll *s) {
  assert(s != NULL &&
      s->groups != NULL);

  time_t now = time(NULL);
  pthread_mutex_lock(&s->groups_lock);
  for (int i = s->groups_count - 1; i >= 0; i--) {
    if (s->groups[i].ts + GROUP_TIMEOUT < now)
      display_gcoll_remove_group(s, i);
  }
  pthread_mutex_unlock(&s->groups_lock);
}

/**
 * IMPORTANT: Lock groups_lock before calling
 */
static int find_group(struct state_gcoll *s, uint32_t ssrc) {
  assert(s != NULL &&
      s->groups != NULL);

  for (int i = 0; i < s->groups_count; i++) {
    if (s->groups[i].ssrc == ssrc)
      return i;
  }

  return -1;
}

struct rum_communicator {
  int                 rap_socket;

  char                local_addr[MAX_ADDRESS_LENGTH + 1];
  char                remote_addr[MAX_ADDRESS_LENGTH + 1];
  int                 port_number; 

  char                *client_msg;
  char                *group_msg;
  char                *stats_msg;

  time_t              last_activity;
  int                 heartbeats;
  bool                failed;
  bool                redirect; 

  struct state_gcoll *gcoll;
};

/**
 * Create client announcement messages, which are periodically sent to reflector.
 * @param r The rum_communicator structure
 * @param p Gcoll parameters in gcoll_init_params structure
 * @return TRUE upon success, FALSE otherwise
 */
static int rum_communicator_create_client_msg(struct rum_communicator *r, struct gcoll_init_params *p) {
  assert(r != NULL &&
      r->client_msg == NULL &&
      r->local_addr != NULL &&
      r->local_addr[0] != '\0' &&
      p != NULL &&
      p->reflector_addr != NULL
      );

  r->client_msg = (char *) calloc(RAP_REQUEST_LENGTH_LIMIT, sizeof (char));
  if (r->client_msg == NULL) {
    fprintf(stderr, "rum_communicator_create_client_msg: Memory allocation error.\n");
    return FALSE;
  }
  snprintf(r->client_msg, RAP_REQUEST_LENGTH_LIMIT,
      "GAZE-BIND RAP/1.0\nTarget: processor/gaze\nIp: %s\nPort: %u\nFront-ssrc: %u\nSide-ssrc: %u \nRoom: %u\n\n",
      r->local_addr, p->port_number, p->front_ssrc, p->side_ssrc, p->group_id);
  return TRUE;
}

/**
 * Create group announcement messages, which are periodically sent to reflector.
 * @param r The rum_communicator structure
 * @param p Gcoll parameters
 * @return TRUE upon success, FALSE otherwise
 */
static int rum_communicator_create_group_msg(struct rum_communicator *r, struct gcoll_init_params *p) {
  assert(r != NULL &&
      r->group_msg == NULL &&
      r->local_addr != NULL &&
      r->local_addr[0] != '\0' &&
      p != NULL &&
      p->send_group_camera &&
      p->reflector_addr != NULL
      );

  r->group_msg = (char *) calloc(RAP_REQUEST_LENGTH_LIMIT, sizeof (char));
  if (r->group_msg == NULL) {
    fprintf(stderr, "rum_communicator_create_group_msg: Memory allocation error.\n");
    return FALSE;
  }
  snprintf(r->group_msg, RAP_REQUEST_LENGTH_LIMIT,
      "GAZE-GROUP RAP/1.0\nTarget: processor/gaze\nIp: %s\nPort: %u\nSsrc: %u\nRoom: %u\n\n",
      r->local_addr, p->port_number, p->group_ssrc, p->group_id);
  return TRUE;
}

/**
 * Create statistics requirement messages, which are periodically sent to reflector.
 * @param r The rum_communicator structure
 * @param p Gcoll parameters
 * @return TRUE upon success, FALSE otherwise
 */
static int rum_communicator_create_stats_msg(struct rum_communicator *r, struct gcoll_init_params *p) {
  assert(r != NULL &&
      r->stats_msg == NULL &&
      p != NULL
      );

  r->stats_msg = (char *) calloc(RAP_REQUEST_LENGTH_LIMIT, sizeof (char));
  if (r->stats_msg == NULL) {
    fprintf(stderr, "rum_communicator_create_stats_msg: Memory allocation error.\n");
    return FALSE;
  }
  snprintf(r->stats_msg, RAP_REQUEST_LENGTH_LIMIT,
      "STAT RAP/1.0\nTarget: processor/gaze\nPort: %u\nStat-type: 1\n\n",
      p->port_number);
  return TRUE;
}

/**
 * Initialize rum_communicator structure
 * @param r The structure
 * @param p Gcoll parameters in gcoll_init_params structure
 * @return TRUE upon success, FALSE otherwise
 */
static int rum_communicator_init(struct rum_communicator *r, struct gcoll_init_params *p, struct state_gcoll *s) {
  assert(r != NULL && p != NULL && p->reflector_addr != NULL);

  r->local_addr[0] = '\0';
  r->port_number = p->port_number;
  r->last_activity = time(NULL);
  r->failed = false;
  r->redirect = false;
  r->gcoll = s;

  if ((r->rap_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    perror("rum_communicator_init socket");
    r->rap_socket = UNDEFINED_FD;
    return FALSE;
  }

  struct hostent *hp;
  if ((hp = gethostbyname((const char *) r->remote_addr)) == NULL) {
    fprintf(stderr, "rum_communicator_init: gethostbyname failed.\n");
    return FALSE;
  }

  int port;
  struct sockaddr_in address;
  port = DEFAULT_RAP_PORT;
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = *(u_long *) hp->h_addr;

  if (connect(r->rap_socket, (struct sockaddr *) &address, sizeof(struct sockaddr)) < 0) {
    perror("rum_communicator_init connect");
    return FALSE;
  }

  struct sockaddr_in name;
  socklen_t namelen = sizeof(name);
  if (getsockname(r->rap_socket, (struct sockaddr *) &name, &namelen) < 0) {
    perror("rum_communicator_init getsockname");
    return FALSE;
  }
  if (inet_ntop(AF_INET, &name.sin_addr, r->local_addr,
        sizeof(r->local_addr) / sizeof(r->local_addr[0]) - 1) == NULL) {
    perror("rum_communicator_init inet_ntop");
    return FALSE;
  }

  if (rum_communicator_create_client_msg(r, p) == FALSE) {
    fprintf(stderr, "Can't create RAP client announcement.\n");
    return FALSE;
  }

  if (p->send_group_camera && rum_communicator_create_group_msg(r, p) == FALSE) {
    fprintf(stderr, "Can't create RAP group announcement.\n");
    return FALSE;
  }

  if (rum_communicator_create_stats_msg(r, p) == FALSE) {
    fprintf(stderr, "Can't create STAT RAP message.\n");
    return FALSE;
  }

  return TRUE;
}

/**
 * Receive reply from reflector after sending a request
 * @param r The rum_communicator structure
 * @param buf Buffer to store the reply, should be at least RAP_REPLY_LENGTH_LIMIT+1 long
 * @return TRUE upon success, FALSE otherwise
 */
static int rum_communicator_get_reply(struct rum_communicator *r, char *buf, unsigned int buflen) {
  int bytes_read;
  int bytes_total = 0;
  char *temp_buf;
  int emptylines = 0;

  assert(r != NULL && buf != NULL && r->rap_socket != UNDEFINED_FD);

  memset((void *) buf, '\0', buflen * sizeof(buf[0]));
  temp_buf = (char *) calloc(buflen, sizeof(char));
  if (temp_buf == NULL) {
    return FALSE;
  }

  do {
    bytes_read = recv(r->rap_socket, temp_buf, RAP_REPLY_LENGTH_LIMIT, 0);
    if (bytes_read < 0) {
      perror("rum_communicator_get_reply read");
      free(temp_buf);
      return FALSE;
    } else if (bytes_read == 0) {
      fprintf(stderr, "rum_communicator_get_reply: read returned 0!");
      free(temp_buf);
      return FALSE;
    } else if (((unsigned int) bytes_read + bytes_total) <= buflen) {
      emptylines = 0;
      strncpy(&buf[bytes_total], temp_buf, bytes_read);
      bytes_total += bytes_read;
      char *emptyline_ptr;
      for (emptyline_ptr = strstr((const char *) buf, "\r\n\r\n"); emptyline_ptr != NULL; emptyline_ptr = strstr((const char *) ++emptyline_ptr, "\r\n\r\n")) {
        emptylines++;
      }
    }
  } while (emptylines < 2);

  free(temp_buf);
  return TRUE;
}

static int rum_communicator_send_msg(struct rum_communicator *r, char *msg) {
  assert(r != NULL && r->rap_socket != UNDEFINED_FD && msg != NULL);

  int buflen = strlen(msg);    
  ssize_t ret = send(r->rap_socket, (const void *) msg, buflen, 0);
  if (ret < 0) {
    perror("rum_communicator_send_msg");
    return FALSE;
  } else if (ret < buflen) {
    fprintf(stderr, "rum_communicator_send_msg: Did not send whole message!");
    return FALSE;
  }

  return TRUE;
}

static bool rum_communicator_parse_stats(struct rum_communicator *r, char *buf) {
  assert(r != NULL && buf != NULL);

  char *line_ptr;
  char *token_ptr;
  char *save_ptr1;
  char *save_ptr2;
  char *temp_str1;
  char *temp_str2;
  int clients_number_parsed = FALSE;
  int groups_number_parsed = FALSE;
  int clients_number;
  int groups_number;
  int magic = 0;
  int magic_line;

  temp_str1 = buf;
  while (1) {
    line_ptr = strtok_r(temp_str1, "\r\n", &save_ptr1);
    if (line_ptr == NULL) break;
    temp_str1 = NULL;
    // We are in "magic"mode
    if (magic > RAP_MAGIC_CODE_START) {
      if (magic == RAP_MAGIC_CODE_REDIRECT && magic_line == 1) {
        char address_str[32];
        char port_str[32];

        sscanf(line_ptr, "%s", (char *) address_str);
        sscanf(line_ptr + strlen(address_str) + 1, "%s", (char *) port_str);
        strcpy(r->remote_addr, address_str);
        break;
      }
      magic_line++;
      continue;
    }
    // Skip message headers. All relevant information is in lines starting with a digit.
    if (!isdigit(line_ptr[0])) {
      continue;
    }
    if (strchr((const char *) line_ptr, ' ') == NULL) { // Current line represents either number of clients or number of groups
      if (clients_number_parsed == FALSE) {
        clients_number = atoi(line_ptr); // this might be slightly weak, but we trust reflector in general
        if (clients_number > RAP_MAGIC_CODE_START) {
          magic = clients_number;
          magic_line = 1;
        }
        if (clients_number < 0 || clients_number > MAX_CLIENTS) {
          fprintf(stderr, "rum_communicator_parse_stats: Bad clients number count: %d\n", clients_number);
          return false;
        }
        clients_number_parsed = TRUE;
      } else {
        groups_number = atoi(line_ptr);
        if (groups_number < 0 || groups_number > MAX_GROUPS) {
          fprintf(stderr, "rum_communicator_parse_stats: Bad groups number count: %d\n", groups_number);
          return false;
        }
        groups_number_parsed = TRUE;
      }
    } else {
      if (groups_number_parsed == TRUE) { // we are parsing information about groups
        uint32_t ssrc = 0;
        int room = 0;
        int i;
        temp_str2 = line_ptr;
        for (i = 1; i <= 2; i++) {
          token_ptr = strtok_r(temp_str2, " ", &save_ptr2);
          if (token_ptr == NULL) break;
          temp_str2 = NULL;
          if (i == 1) {
            ssrc = (uint32_t) atol(token_ptr);
            continue;
          }
          if (i == 2) {
            room = atoi(token_ptr);
            continue;
          }
        }
        if (ssrc != 0 && ssrc != r->gcoll->params->group_ssrc)
          add_or_update_group(r->gcoll, ssrc, room);
      } else { // we are parsing information about clients
        uint32_t ssrc = 0;
        uint32_t gaze_ssrc = 0;
        uint32_t side_ssrc = 0;
        int room = 0;
        int i;
        temp_str2 = line_ptr;
        for (i = 1; i <= 4; i++) {
          token_ptr = strtok_r(temp_str2, " ", &save_ptr2);
          if (token_ptr == NULL) break;
          temp_str2 = NULL;
          switch (i) {
            case 1:
              ssrc = (uint32_t) atol(token_ptr);
              break;
            case 2:
              side_ssrc = (uint32_t) atol(token_ptr);
              break;
            case 3:
              gaze_ssrc = (uint32_t) atol(token_ptr);
              break;
            case 4:
              room = atoi(token_ptr);
              break;
            default:
              break;
          }
        }
        if (ssrc != 0 && ssrc != r->gcoll->params->front_ssrc)
          add_or_update_participant(r->gcoll, ssrc, side_ssrc, gaze_ssrc, room);
      }
    }
  }

  return true;
}

static void rum_communicator_build_gaze_msg(struct rum_communicator *r, char *buf) {
  assert(r != NULL &&
      r->gcoll != NULL &&
      r->gcoll->params != NULL &&
      buf != NULL);

  sprintf(buf, "GAZE RAP/1.0\nTarget: processor/gaze\nPort: %u\nGaze-who: %u\nGaze-at: %u\n\n",
      r->port_number, r->gcoll->params->front_ssrc, r->gcoll->gaze_ssrc);
}

static void rum_communicator_heartbeat(struct rum_communicator *r) {
  assert(r != NULL && r->rap_socket != UNDEFINED_FD);

  char reply_buf[RAP_REPLY_LENGTH_LIMIT + 1];
  char stats_buf[4 * RAP_REPLY_LENGTH_LIMIT];

  if (r->gcoll->gaze_changed) {
    char gaze_msg[RAP_REQUEST_LENGTH_LIMIT + 1];
    rum_communicator_build_gaze_msg(r, gaze_msg);
    if (! (rum_communicator_send_msg(r, gaze_msg) == TRUE &&
          rum_communicator_get_reply(r, reply_buf, sizeof(reply_buf) / sizeof(reply_buf[0])) == TRUE)) {
      r->failed = true;
      return;
    }
    r->gcoll->gaze_changed = false;
  }

  if (r->client_msg != NULL) {
    if (! (rum_communicator_send_msg(r, r->client_msg) == TRUE &&
          rum_communicator_get_reply(r, reply_buf, sizeof(reply_buf) / sizeof(reply_buf[0])) == TRUE)) {
      r->failed = true;
      return;
    }
  }

  if (r->group_msg != NULL) {
    if (! (rum_communicator_send_msg(r, r->group_msg) == TRUE &&
          rum_communicator_get_reply(r, reply_buf, sizeof(reply_buf) / sizeof(reply_buf[0])) == TRUE)) {
      r->failed = true;
      return;
    }
  }

  if (r->stats_msg != NULL) {
    if (!(rum_communicator_send_msg(r, r->stats_msg) == TRUE &&
          rum_communicator_get_reply(r, stats_buf, sizeof(stats_buf) / sizeof(stats_buf[0])) == TRUE)) {
      r->failed = true;
      return;
    }
    if (rum_communicator_parse_stats(r, stats_buf) == FALSE) {
      r->failed = true;
      return;
    }
  }

  if (r->remote_addr[0] != '\0') {
  }

  r->last_activity = time(NULL);
}

/**
 * Clean up rum_communicator structure
 * @param r The structure
 */
static void rum_communicator_done(struct rum_communicator *r) {
  assert(r != NULL);

  close(r->rap_socket);
  r->rap_socket = UNDEFINED_FD;

  free(r->client_msg);
  free(r->group_msg);
  free(r->stats_msg);

  r->client_msg = r->group_msg = r->stats_msg = NULL;
}

static void * rum_communicator_main(void *s) {
  assert(s != NULL);

  struct state_gcoll *sg = (struct state_gcoll *) s;

  while (!sg->exit) {
    if (!sg->rum->redirect) {
      strcpy(sg->rum->remote_addr, sg->params->reflector_addr);
    }

    if (rum_communicator_init(sg->rum, sg->params, sg) == FALSE) {
      fprintf(stderr, "RUM communicator initialization failed.\n");
    fflush(stderr);
      sg->exit = true;
      return NULL;
    }

    if (sg->rum->redirect) {
      int retval = TRUE;

      for (int i = 0; i < CAP_DEV_COUNT; i++) {
        retval = retval && rtp_set_new_addr(sg->params->rtp_session[i], sg->rum->remote_addr);
      }
      if (retval == FALSE) {
        fprintf(stderr, "New sending addresses could not be set, quitting.");
    fflush(stderr);
        sg->exit = true;
      }
    }

    sg->rum->heartbeats = 0;
    while (!sg->rum->failed && !sg->rum->redirect && !sg->exit) {
      rum_communicator_heartbeat(sg->rum);
      sg->rum->heartbeats++;
      usleep(500 * 1000);
    }
    if (sg->rum->heartbeats < 2) {
      fprintf(stderr, "Frequent RUM communicator failures, quitting.\n");
    fflush(stderr);
      sg->exit = true;
    }

    rum_communicator_done(sg->rum);
  }

  return NULL;
}

static void gl_check_error()
{
  GLenum msg;
  int flag=0;
  msg=glGetError();
  while(msg!=GL_NO_ERROR) {
    flag=1;
    switch(msg){
      case GL_INVALID_ENUM:
        fprintf(stderr, "GL_INVALID_ENUM\n");
        break;
      case GL_INVALID_VALUE:
        fprintf(stderr, "GL_INVALID_VALUE\n");
        break;
      case GL_INVALID_OPERATION:
        fprintf(stderr, "GL_INVALID_OPERATION\n");
        break;
      case GL_STACK_OVERFLOW:
        fprintf(stderr, "GL_STACK_OVERFLOW\n");
        break;
      case GL_STACK_UNDERFLOW:
        fprintf(stderr, "GL_STACK_UNDERFLOW\n");
        break;
      case GL_OUT_OF_MEMORY:
        fprintf(stderr, "GL_OUT_OF_MEMORY\n");
        break;
      case 1286:
        fprintf(stderr, "INVALID_FRAMEBUFFER_OPERATION_EXT\n");
        break;
      default:
        fprintf(stderr, "wft mate? Unknown GL ERROR: %d\n", msg);
        break;
    }
    msg=glGetError();
  }
  if(flag)
    abort();
}

static void setup_texture_parameters(void) {
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

static void reset_frame_storage(struct state_gcoll *s) {
  s->current_frames->count = 2 * s->participants_count + s->groups_count;
  for (int i = 0; i < s->participants_count; i++) {
    s->current_frames->ssrc[i] = s->participants[i].ssrc;
    s->current_frames->frames[i] = NULL;
  }
  for (int i = 0; i < s->participants_count; i++) {
    s->current_frames->ssrc[i + s->participants_count] = s->participants[i].side_ssrc;
    s->current_frames->frames[i + s->participants_count] = NULL;
  }
  for (int i = 0; i < s->groups_count; i++) {
    s->current_frames->ssrc[2 * s->participants_count + i] = s->groups[i].ssrc;
    s->current_frames->frames[2 * s->participants_count + i] = NULL;
  }
}

static void glut_idle_callback(void) {
  struct state_gcoll *s = gcoll;

  s->calls++;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double seconds = tv_diff(tv, s->tv);
  if (seconds > 5) {
    double cps = s->calls / seconds;
    s->calls = 0;
    s->tv = tv;
  }

  pthread_mutex_lock(&s->new_frames_lock);
  // Quickly check if there is any new frame; If it is not, we do not need to manipulate
  // the whole frame_storage structure
  if (!s->received_frame) {
    bool change_reported = false;
    pthread_mutex_lock(&s->participants_lock);
    if (s->participants_modified) {
      change_reported = true;
      s->participants_modified = false;
    }
    pthread_mutex_unlock(&s->participants_lock);
    pthread_mutex_lock(&s->groups_lock);
    if (s->groups_modified) {
      change_reported = true;
      s->groups_modified = false;
    }
    pthread_mutex_unlock(&s->groups_lock);
    if (change_reported) {
      reset_frame_storage(s);
      struct frame_storage *tmp = s->current_frames;
      s->current_frames = s->new_frames;
      s->new_frames = tmp;
    }

    pthread_mutex_unlock(&s->new_frames_lock);
    return;
  }

  // Get new frames from the data receiving thread
  reset_frame_storage(s);
  {
    struct frame_storage *tmp = s->current_frames;
    s->current_frames = s->new_frames;
    s->new_frames = tmp;
  }
  s->received_frame = false;
  pthread_mutex_unlock(&s->new_frames_lock);

  // Update new users; 
  pthread_mutex_lock(&s->participants_lock);
  pthread_mutex_lock(&s->groups_lock);
  for (int i = 0; i < s->current_frames->count; i++) {
    if ((s->current_frames->frames[i] == NULL && s->participants[i].ssrc == s->gaze_ssrc))
      continue;

    int id = find_participant(s, s->current_frames->ssrc[i]);
    if (id >= 0) {
      vf_free_data(s->participants[id].frame);
      s->participants[id].frame = s->current_frames->frames[i];
      participant_update_ts(&s->participants[id]);
      continue;
    }
    id = find_participant_side(s, s->current_frames->ssrc[i]);
    if (id >= 0) {
      vf_free_data(s->participants[id].side_frame);
      s->participants[id].side_frame = s->current_frames->frames[i];
      participant_update_ts(&s->participants[id]);
      continue;
    }
    id = find_group(s, s->current_frames->ssrc[i]);
    if (id >= 0) {
      vf_free_data(s->groups[id].frame);
      s->groups[id].frame = s->current_frames->frames[i];
      group_update_ts(&s->groups[id]);
      continue;
    }

    // SSRC disappeared
    vf_free_data(s->current_frames->frames[i]);
  }
  pthread_mutex_unlock(&s->participants_lock);
  pthread_mutex_unlock(&s->groups_lock);

  int screen_width = glutGet(GLUT_WINDOW_WIDTH);
  int screen_height = glutGet(GLUT_WINDOW_HEIGHT);
  float maximum_small_width = 1.0 / (float) s->participants_count - 0.01;
  float maximum_small_height = 0.4;
  maximum_small_width = min(maximum_small_width, maximum_small_height);
  float small_win_ratio = min(maximum_small_width, maximum_small_height);
  float screen_ratio = (float) screen_width / screen_height;

  float width_bound = maximum_small_width * screen_width;
  float height_bound = maximum_small_height * screen_height;
  float bound_ratio = width_bound / height_bound;

  glClear(GL_COLOR_BUFFER_BIT);

  gl_check_error();

  pthread_mutex_lock(&s->participants_lock);
  s->small_images_count = s->participants_count;
  float gap_width = (float) (1 - maximum_small_width * s->participants_count) / (s->participants_count + 1);
  float left = -1.0 +  2 * gap_width;
  for (int i = 0; i < s->participants_count; i++) {
    //if ((s->participants[i].frame == NULL && s->participants[i].ssrc == s->gaze_ssrc)) continue;

    if (s->participants[i].gaze_ssrc == s->params->front_ssrc && s->participants[i].frame != NULL) {
    if (s->participants[i].texture == 0) {
      glGenTextures(1, &s->participants[i].texture);
      gl_check_error();
      glBindTexture(GL_TEXTURE_2D, s->participants[i].texture);
      gl_check_error();
      setup_texture_parameters();
      gl_check_error();
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
          s->participants[i].frame->tiles[0].width,
          s->participants[i].frame->tiles[0].height,
          0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
      gl_check_error();
      fprintf(stderr, "generate texture: %d\n", i);
    } else {
      glBindTexture(GL_TEXTURE_2D, s->participants[i].texture);
    }

    gl_check_error();

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
        s->participants[i].frame->tiles[0].width,
        s->participants[i].frame->tiles[0].height,
        GL_RGB, GL_UNSIGNED_BYTE,
        s->participants[i].frame->tiles[0].data);

    gl_check_error();

    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.35f);

    gl_check_error();

    float small_height = 0.0;
    float small_width = 0.0;
    float frame_ratio = (float) s->participants[i].frame->tiles[0].width /
      s->participants[i].frame->tiles[0].height;
    if (s->participants[i].ssrc != s->gaze_ssrc) {
      float right = left + 2 * maximum_small_width;
      float bottom = -1.0;
      float top = -1.0 + 2 * maximum_small_height;

  //fprintf(stderr, "%f %f\n", bound_ratio, frame_ratio);
      if (bound_ratio > frame_ratio) {
        float center = (left + right) / 2;
        float win_width = (right - left) * frame_ratio * (float) screen_height / screen_width;
        left = center - win_width / 2;
        right = center + win_width / 2;
      } else {
        float center = (bottom + top) / 2;
        float win_height = (top - bottom) / frame_ratio * (float) screen_width / screen_height;
        bottom = center - win_height / 2;
        top = center + win_height / 2;
      }
      glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 1.0f); glVertex2f(left, bottom);
      glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
      glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
      glTexCoord2f(0.0f, 0.0f); glVertex2f(left, top);
      glEnd();
    } else {
      float top = 1.0;
      float left = 0.0;
      float bottom = 0.0;
      float right = 1.0;
      if (screen_ratio > frame_ratio) {
        float big_width = (right - left) * (float) s->participants[i].frame->tiles[0].width / 
          s->participants[i].frame->tiles[0].height *
          (float) screen_height / screen_width;
        left = 0.5 - big_width / 2;
        right = 0.5 + big_width / 2;
      } else {
        float big_height = (top - bottom) * (float) s->participants[i].frame->tiles[0].height / 
          s->participants[i].frame->tiles[0].width *
          (float) screen_width / screen_height;
        bottom = 0.5 - big_height / 2;
        top = 0.5 + big_height / 2;
      }
      glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 1.0f); glVertex2f(left, bottom);
      glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
      glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
      glTexCoord2f(0.0f, 0.0f); glVertex2f(left, top);
      glEnd();
    }
} else 
    if (s->participants[i].gaze_ssrc != s->params->front_ssrc && s->participants[i].side_frame != NULL) {
    if (s->participants[i].side_texture == 0) {
      glGenTextures(1, &s->participants[i].side_texture);
      gl_check_error();
      glBindTexture(GL_TEXTURE_2D, s->participants[i].side_texture);
      gl_check_error();
      setup_texture_parameters();
      gl_check_error();
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
          s->participants[i].side_frame->tiles[0].width,
          s->participants[i].side_frame->tiles[0].height,
          0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
      gl_check_error();
      fprintf(stderr, "generate texture: %d\n", i);
    } else {
      glBindTexture(GL_TEXTURE_2D, s->participants[i].side_texture);
    }

    gl_check_error();

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
        s->participants[i].side_frame->tiles[0].width,
        s->participants[i].side_frame->tiles[0].height,
        GL_RGB, GL_UNSIGNED_BYTE,
        s->participants[i].side_frame->tiles[0].data);

    gl_check_error();

    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.35f);

    gl_check_error();

    float small_height = 0.0;
    float small_width = 0.0;
    float frame_ratio = (float) s->participants[i].side_frame->tiles[0].width /
      s->participants[i].side_frame->tiles[0].height;
    if (s->participants[i].ssrc != s->gaze_ssrc) {
      float right = left + 2 * maximum_small_width;
      float bottom = -1.0;
      float top = -1.0 + 2 * maximum_small_height;

  //fprintf(stderr, "%f %f\n", bound_ratio, frame_ratio);
      if (bound_ratio > frame_ratio) {
        float center = (left + right) / 2;
        float win_width = (right - left) * frame_ratio * (float) screen_height / screen_width;
        left = center - win_width / 2;
        right = center + win_width / 2;
      } else {
        float center = (bottom + top) / 2;
        float win_height = (top - bottom) / frame_ratio * (float) screen_width / screen_height;
        bottom = center - win_height / 2;
        top = center + win_height / 2;
      }
      glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 1.0f); glVertex2f(left, bottom);
      glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
      glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
      glTexCoord2f(0.0f, 0.0f); glVertex2f(left, top);
      glEnd();
    } else {
      float top = 1.0;
      float left = 0.0;
      float bottom = 0.0;
      float right = 1.0;
      if (screen_ratio > frame_ratio) {
        float big_width = (right - left) * (float) s->participants[i].side_frame->tiles[0].width / 
          s->participants[i].side_frame->tiles[0].height *
          (float) screen_height / screen_width;
        left = 0.5 - big_width / 2;
        right = 0.5 + big_width / 2;
      } else {
        float big_height = (top - bottom) * (float) s->participants[i].side_frame->tiles[0].height / 
          s->participants[i].side_frame->tiles[0].width *
          (float) screen_width / screen_height;
        bottom = 0.5 - big_height / 2;
        top = 0.5 + big_height / 2;
      }
      glBegin(GL_QUADS);
      glTexCoord2f(0.0f, 1.0f); glVertex2f(left, bottom);
      glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
      glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
      glTexCoord2f(0.0f, 0.0f); glVertex2f(left, top);
      glEnd();
    }
}

    gl_check_error();
    left += 2 * (maximum_small_width + gap_width);

  }
  pthread_mutex_unlock(&s->participants_lock);

  // Just one group window now...
  pthread_mutex_lock(&s->groups_lock);
  for (int i = 0; i < s->groups_count; i++)
  {
    if (s->groups[i].frame == NULL) continue;
    float frame_ratio = (float) s->groups[i].frame->tiles[0].width /
      s->groups[i].frame->tiles[0].height;
    float top = 1.0;
    float left = -1.0;
    float bottom = 0.0;
    float right = 0.0;
    if (screen_ratio > frame_ratio) {
      float big_width = (right - left) * (float) s->groups[i].frame->tiles[0].width / 
        s->groups[i].frame->tiles[0].height *
        (float) screen_height / screen_width;
      left = -0.5 - big_width / 2;
      right = -0.5 + big_width / 2;
    } else {
      float big_height = (top - bottom) * (float) s->groups[i].frame->tiles[0].height / 
        s->groups[i].frame->tiles[0].width *
        (float) screen_width / screen_height;
      bottom = 0.5 - big_height / 2;
      top = 0.5 + big_height / 2;
    }
    if (s->groups[i].texture == 0) {
      glGenTextures(1, &s->groups[i].texture);
      glBindTexture(GL_TEXTURE_2D, s->groups[i].texture);
      setup_texture_parameters();
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
          s->groups[i].frame->tiles[0].width,
          s->groups[i].frame->tiles[0].height,
          0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    } else {
      glBindTexture(GL_TEXTURE_2D, s->groups[i].texture);
    }

    gl_check_error();

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
        s->groups[i].frame->tiles[0].width,
        s->groups[i].frame->tiles[0].height,
        GL_RGB, GL_UNSIGNED_BYTE,
        s->groups[i].frame->tiles[0].data);

    gl_check_error();

    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -1.35f);

    gl_check_error();
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(left, bottom);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(right, bottom);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(right, top);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(left, top);
    glEnd();

    gl_check_error();

  }
  pthread_mutex_unlock(&s->groups_lock);

  glutPostRedisplay();
}

static void clicked_coords(int x, int y) {
  struct state_gcoll *s = gcoll;

  int width = glutGet(GLUT_WINDOW_WIDTH);
  int height = glutGet(GLUT_WINDOW_HEIGHT);

  fprintf(stderr, "clicked: %d %d\n", x, y);
  float yr = ((float) y) / height;
  float xr = ((float ) x) / width;

  // Top part with big images
  if (yr < 0.6) {
    s->gaze_ssrc = 0;
    s->gaze_changed = true;
  } else {
    if (s->small_images_count == 0) return;

    int order = floor(xr * s->small_images_count);
    if (s->gaze_ssrc != s->current_frames->ssrc[order]) {
      s->gaze_ssrc = s->current_frames->ssrc[order]; // this should be participant`s ssrc
      s->gaze_changed = true;
    }
  }
  fprintf(stderr, "relative: %f %f\n", xr, yr);
}

static void glut_key_callback(unsigned char key, int x, int y)
{
  UNUSED(x);
  UNUSED(y);

  switch(key) {
    case 'q':
      if(gcoll->glut_window != -1)
        glutDestroyWindow(gcoll->glut_window);
      exit_uv(0);
      break;
  }
}

static void glut_mouse_callback(int button, int state, int x, int y) {
  UNUSED(button);

  if (state == GLUT_DOWN) {
    clicked_coords(x, y);
  }
}

void *display_gcoll_init(char *fmt, unsigned int flags, void *udata)
{
  UNUSED(fmt);
  UNUSED(flags);

  struct state_gcoll *s;

  s = (struct state_gcoll *)calloc(1, sizeof(struct state_gcoll));
  if (s == NULL) {
    fprintf(stderr, "display_gcoll_init: Memory allocation error.\n");
    return NULL;
  }

  // initialize static pointer
  gcoll = s;

  s->params = (struct gcoll_init_params *) udata;
  s->gaze_ssrc = 0;
  s->gaze_changed = false;
  s->small_images_count = 0;
  s->new_frames = calloc(1, sizeof(struct frame_storage));
  s->current_frames = calloc(1, sizeof(struct frame_storage));
  s->received_frame = false;
  s->exit = false;
  s->magic = MAGIC_GCOLL;

  s->participants = (struct participant *) calloc(MAX_CLIENTS, sizeof(struct participant));
  s->groups = (struct group *) calloc(MAX_GROUPS, sizeof(struct group));
  s->rum = (struct rum_communicator *) calloc(1, sizeof(struct rum_communicator));
  if (s->participants == NULL ||
      s->groups == NULL || 
      s->rum == NULL) {
    fprintf(stderr, "display_gcoll_init: Memory allocation error.\n");
    goto error;
  }

  for (int i = 0; i < MAX_CLIENTS; i++) participant_init(&s->participants[i]);
  for (int i = 0; i < MAX_GROUPS; i++) group_init(&s->groups[i]);
  s->participants_count = s->groups_count = 0;
  s->participants_modified = s->groups_modified = false;

  pthread_mutex_init(&s->participants_lock, NULL);
  pthread_mutex_init(&s->groups_lock, NULL);
  pthread_mutex_init(&s->new_frames_lock, NULL);

  glutInit(&uv_argc, uv_argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

  glutIdleFunc(glut_idle_callback);
  s->glut_window = glutCreateWindow(WIN_NAME);
  glutSetCursor(GLUT_CURSOR_INHERIT);
  glutKeyboardFunc(glut_key_callback);
  glutMouseFunc(glut_mouse_callback);
  glutDisplayFunc(glutSwapBuffers);

#ifdef HAVE_LINUX
  GLenum err;
  if ((err = glewInit()) != GLEW_OK)
  {
    /* Problem: glewInit failed, something is seriously wrong. */
    fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
    goto error;
  }
#endif /* HAVE_LINUX */

  glutShowWindow();
  glutFullScreen();
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glEnable(GL_TEXTURE_2D);

  int width = glutGet(GLUT_WINDOW_WIDTH);
  int height = glutGet(GLUT_WINDOW_HEIGHT);

  glViewport(0, 0, (GLint) width, (GLint) height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-1, 1, -1, 1, 10, -10);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glClear(GL_COLOR_BUFFER_BIT);
  glutPostRedisplay();

  gl_check_error();

  if (pthread_create(&s->rum_thread, NULL, &rum_communicator_main, (void *) s)) {
    fprintf(stderr, "Cannot create RUM communicator thread.\n");
    goto error;
  }

  s->calls = 0ul;
  gettimeofday(&s->tv, NULL);

  return (void *) s;

error:
  free(s);
  return NULL;
}

void display_gcoll_run(void *arg)
{
  UNUSED(arg);

  struct state_gcoll *s = gcoll;

  while (!gcoll->exit) {
    glutMainLoopEvent();
    glut_idle_callback();
    gl_check_error();

    remove_inactive_participants(s);
    remove_inactive_groups(s);
  }

}

void display_gcoll_finish(void *state)
{
  UNUSED(state);
}

void display_gcoll_done(void *state)
{
  assert(state != NULL);
  struct state_gcoll *s = (struct state_gcoll *)state;

  s->exit = true;
  pthread_join(s->rum_thread, NULL);
  rum_communicator_done(s->rum);
  s->rum = NULL;

  free(s->participants);
  free(s->groups);
  s->participants = NULL;
  s->groups = NULL;

  pthread_mutex_destroy(&s->participants_lock);
  pthread_mutex_destroy(&s->groups_lock);

  assert(s->magic == MAGIC_GCOLL);
  free(s);
}

int display_gcoll_putf(void *state, struct video_frame *frame)
{
  assert(state != NULL);
  struct state_gcoll *s = (struct state_gcoll *)state;
  assert(s->magic == MAGIC_GCOLL);

//fprintf(stderr, "frame: %d\n", frame->ssrc);
  /* TODO: might this hapen? */
  if (frame == NULL) return 0;

  /*
     int id;

     pthread_mutex_lock(&s->participants_lock);
     id = find_participant(s, frame->ssrc);
     if (id >= 0) {
     vf_free_data(s->participants[id]->frame);
     s->participants[id]->frame = frame;
     pthread_mutex_unlock(&s->participants_lock);
     s->new_frame = true;
     return 0;
     }
     pthread_mutex_unlock(&s->participants_lock);

     pthread_mutex_lock(&s->groups_lock);
     id = find_group(s, frame->ssrc);
     if (id >= 0) {
     vf_free_data(s->groups[id]->frame);
     s->groups[id]->frame = frame;
     pthread_mutex_unlock(&s->groups_lock);
     s->new_frame = true;
     return 0;
     }
     pthread_mutex_unlock(&s->groups_lock);
     */

  pthread_mutex_lock(&s->new_frames_lock);
//fprintf(stderr, "count: %d\n", s->new_frames->count);
  for (int i = 0; i < s->new_frames->count; i++) {
//fprintf(stderr, "ssrc: %d\n", s->new_frames->ssrc[i]);
    if (frame->ssrc == s->new_frames->ssrc[i]) {
      s->new_frames->frames[i] = frame;
      s->received_frame = true;
      pthread_mutex_unlock(&s->new_frames_lock);
      return 0;
    }
  }
  pthread_mutex_unlock(&s->new_frames_lock);

  vf_free_data((struct video_frame *)frame);

  return 0;
}

display_type_t *display_gcoll_probe(void)
{
  display_type_t *dt;

  dt = malloc(sizeof(display_type_t));
  if (dt != NULL) {
    dt->id = DISPLAY_GCOLL_ID;
    dt->name = "gcoll";
    dt->description = "GColl display device";
  }
  return dt;
}

int display_gcoll_get_property(void *state, int property, void *val, size_t *len)
{
  UNUSED(state);
  codec_t codecs[] = { RGB };
  enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};

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
      *(int *) val = 0;
      *len = sizeof(int);
      break;
    case DISPLAY_PROPERTY_GSHIFT:
      *(int *) val = 8;
      *len = sizeof(int);
      break;
    case DISPLAY_PROPERTY_BSHIFT:
      *(int *) val = 16;
      *len = sizeof(int);
      break;
    case DISPLAY_PROPERTY_BUF_PITCH:
      *(int *) val = PITCH_DEFAULT;
      *len = sizeof(int);
      break;
    case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
      if(sizeof(supported_il_modes) <= *len) {
        memcpy(val, supported_il_modes, sizeof(supported_il_modes));
      } else {
        return FALSE;
      }
      *len = sizeof(supported_il_modes);
      break;
    default:
      return FALSE;
  }
  return TRUE;
}

