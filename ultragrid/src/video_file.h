#ifndef video_file_h
#define video_file_h

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <string>

#include "video.h"

enum class filetype {
        FILETYPE_J2K
};

class video_file {
        public:
                virtual ~video_file() {}
                virtual struct video_desc get_video_desc() = 0;
                /**
                 * Returns raw data that will be transmitted with SW
                 *
                 * May be either with header (opaque compressed images such as J2K)
                 * or without (DPX).
                 *
                 * @param[out] len  data lenght
                 * @return          actual data
                 *
                 */
                virtual char *get_raw_data(int &len) = 0;

                static video_file *create(enum filetype type, std::string name);
                static enum filetype get_filetype_to_ext(std::string extension);
};

#endif // video_file_h

