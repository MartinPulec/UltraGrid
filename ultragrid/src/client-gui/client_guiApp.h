/***************************************************************
 * Name:      client_guiApp.h
 * Purpose:   Defines Application Class
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#ifndef CLIENT_GUIAPP_H
#define CLIENT_GUIAPP_H

#include <wx/app.h>

class client_guiApp : public wxApp
{
    public:
        virtual bool OnInit();
};

#endif // CLIENT_GUIAPP_H
