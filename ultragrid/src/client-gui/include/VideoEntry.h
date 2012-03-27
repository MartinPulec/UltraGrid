#ifndef VIDEOENTRY_H
#define VIDEOENTRY_H

#include <wx/string.h>
#include <wx/dynarray.h>

enum colorSpace {
    RGB_YCbCr_709_D65,
    XYZ
};

struct VideoEntry
{
    public:
        VideoEntry();
        virtual ~VideoEntry();

        wxString URL;
        wxString format;
        double fps;
        long int total_frames;

        wxString colorSpace;

    protected:
    private:
};

WX_DECLARE_OBJARRAY(VideoEntry, ArrayOfVideoEntries);


#endif // VIDEOENTRY_H
