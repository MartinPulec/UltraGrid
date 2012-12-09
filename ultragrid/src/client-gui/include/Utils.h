#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <netinet/in.h>
#include <wx/string.h>

#include "video.h"

class Utils
{
    public:
        Utils();
        virtual ~Utils();

        static wxString FromCDouble(double value, int precision);
        static bool boolFromString(std::string);

        static int conn_nonb(struct sockaddr_in sa, int sock, int timeout);

        static void toV210(char *src, char *dst, int width, int height);
        static void scale(int sw, int sh, int src_pitch_pix, int *src, int dw, int dh, int *dst);

        static std::string VideoDescSerialize(struct video_desc *mode);
        static struct video_desc VideoDescDeserialize(std::string modeStr);
    protected:
    private:
};

#endif // UTILS_H
