/***************************************************************
 * Name:      serv_guiApp.cpp
 * Purpose:   Code for Application Class
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#include "serv_guiApp.h"

//(*AppHeaders
#include "serv_guiMain.h"
#include <wx/image.h>
//*)

IMPLEMENT_APP(serv_guiApp);

bool serv_guiApp::OnInit()
{
    //(*AppInitialize
    bool wxsOK = true;
    wxInitAllImageHandlers();
    if ( wxsOK )
    {
    	serv_guiFrame* Frame = new serv_guiFrame(0);
    	Frame->Show();
    	SetTopWindow(Frame);
    }
    //*)
    return wxsOK;

}
