#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video_file.h"

class j2k_file : public video_file {
        public:
                j2k_file(std::string name);
                ~j2k_file();
                struct video_desc get_video_desc();
                char *get_raw_data(int &len);
        private:
                std::string m_name;
};

