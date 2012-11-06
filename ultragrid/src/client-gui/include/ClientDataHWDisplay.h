#ifndef CLIENTDATAHWDISPLAY_H
#define CLIENTDATAHWDISPLAY_H

#include <wx/clntdata.h>

extern "C" struct video_desc;

struct ClientDataHWDisplay: public wxClientData
{
    ClientDataHWDisplay(const char *identifier, struct video_desc *descs, ssize_t desc_count);
    virtual ~ClientDataHWDisplay();

    char               *identifier;
    struct video_desc  *modes;
    ssize_t              modes_count;
};

#endif // CLIENTDATAHWDISPLAY_H
