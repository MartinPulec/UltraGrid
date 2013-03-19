#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video_file.h"

#include <algorithm>
#include <stdexcept>

#include "video_file/j2k.h"

using namespace std;

video_file * video_file::create(enum filetype type, std::string name)
{
        switch(type) {
                case filetype::FILETYPE_J2K:
                        return new j2k_file(name);
                default:
                        throw logic_error("Unknown filetype");
        }
}

enum filetype video_file::get_filetype_to_ext(string ext)
{
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if(ext.compare("j2k") == 0) {
                return filetype::FILETYPE_J2K;
        } else {
                throw runtime_error("Unknown extension");
        }

}

