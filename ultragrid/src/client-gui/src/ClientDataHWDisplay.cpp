#include "ClientDataHWDisplay.h"

#include "video.h"

ClientDataHWDisplay::ClientDataHWDisplay(const char *identifier, struct video_desc *descs, ssize_t desc_count)
{
    this->identifier = strdup(identifier);
    this->modes = (struct video_desc *) malloc(sizeof(struct video_desc) * desc_count);
    if(desc_count > 0) {
        memcpy(this->modes, descs, sizeof(struct video_desc) * desc_count);
    }
    this->modes_count = desc_count;
}

ClientDataHWDisplay::~ClientDataHWDisplay()
{
    free(this->identifier);
    free(this->modes);
}
