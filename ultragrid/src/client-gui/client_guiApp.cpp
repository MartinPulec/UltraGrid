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
#include <iostream>

#include "video_display.h"

//(*AppHeaders
#include "client_guiMain.h"
#include <wx/image.h>
//*)

IMPLEMENT_APP(client_guiApp);

static client_guiFrame *app = NULL;

bool client_guiApp::OnInit()
{
    signal(SIGPIPE, SIG_IGN);

    if (display_init_devices() != 0) {
        fprintf(stderr, "Unable to initialise devices\n");
        abort();
    } else {
        printf("Found %d display devices\n",
                  display_get_device_count());
    }

    //(*AppInitialize
    bool wxsOK = true;
    wxInitAllImageHandlers();
    if ( wxsOK )
    {
    	client_guiFrame* Frame = new client_guiFrame(0);
    	Frame->SetAutoLayout(true);
    	Frame->Show();
    	SetTopWindow(Frame);
    	app = Frame;
    }
    //*)

    return wxsOK;

}

int client_guiApp::OnExit()
{
    display_free_devices();

    return 0;
}
