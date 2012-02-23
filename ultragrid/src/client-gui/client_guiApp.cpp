/***************************************************************
 * Name:      client_guiApp.cpp
 * Purpose:   Code for Application Class
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#include "client_guiApp.h"
#include <signal.h>

//(*AppHeaders
#include "client_guiMain.h"
#include <wx/image.h>
//*)

IMPLEMENT_APP(client_guiApp);

bool client_guiApp::OnInit()
{
    //(*AppInitialize
    bool wxsOK = true;
    wxInitAllImageHandlers();
    if ( wxsOK )
    {
    	client_guiFrame* Frame = new client_guiFrame(0);
    	Frame->Show();
    	SetTopWindow(Frame);
    }
    //*)

    signal(SIGPIPE, SIG_IGN);
    return wxsOK;

}
