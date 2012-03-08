/***************************************************************
 * Name:      serv_guiApp.h
 * Purpose:   Defines Application Class
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#ifndef SERV_GUIAPP_H
#define SERV_GUIAPP_H

#include <wx/app.h>

class serv_guiApp : public wxApp
{
    public:
        virtual bool OnInit();
};

#endif // SERV_GUIAPP_H
