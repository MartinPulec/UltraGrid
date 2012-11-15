#ifndef VIDEOENTRY_H
#define VIDEOENTRY_H

#include <wx/string.h>
#include <wx/dynarray.h>

extern const char *possibleFileFormats[];
extern int possibleFileFormatsCount;

struct VideoEntry
{
    public:
        VideoEntry();
        virtual ~VideoEntry();

        VideoEntry(wxString & encoded);

        wxString Serialize();

        wxString URL;
        wxString format;
        double fps;
        long int total_frames;

        wxString colorSpace;

        wxString audioFile;

    protected:
    private:
};

WX_DECLARE_OBJARRAY(VideoEntry, ArrayOfVideoEntries);


#endif // VIDEOENTRY_H
