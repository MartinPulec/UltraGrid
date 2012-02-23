/***************************************************************
 * Name:      client_guiMain.h
 * Purpose:   Defines Application Frame
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#ifndef CLIENT_GUIMAIN_H
#define CLIENT_GUIMAIN_H

#include "include/sp_client.h"
#include "include/UltraGridManager.h"
#include "include/Settings.h"

//(*Headers(client_guiFrame)
#include <wx/sizer.h>
#include <wx/button.h>
#include <wx/menu.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/listbox.h>
//*)

class client_guiFrame: public wxFrame
{
    public:

        client_guiFrame(wxWindow* parent,wxWindowID id = -1);
        virtual ~client_guiFrame();

    private:

        //(*Handlers(client_guiFrame)
        void OnQuit(wxCommandEvent& event);
        void OnAbout(wxCommandEvent& event);
        void OnServerSetting(wxCommandEvent& event);
        void OnListBox1Select(wxCommandEvent& event);
        void OnListBox1Select1(wxCommandEvent& event);
        void OnButton1Click(wxCommandEvent& event);
        void OnPlayClick(wxCommandEvent& event);
        //*)

        void PlaySelection();
        void NotifyWindowClosed();
        void SetPlayButtonStop(bool stop);

        //(*Identifiers(client_guiFrame)
        static const long ID_LISTBOX1;
        static const long ID_BUTTON1;
        static const long PlayButton;
        static const long idMenuQuit;
        static const long idServerSetting;
        static const long idMenuAbout;
        static const long ID_STATUSBAR1;
        //*)

        //(*Declarations(client_guiFrame)
        wxListBox* ListBox1;
        wxStatusBar* StatusBar1;
        wxButton* Button1;
        wxButton* Play;
        //*)

        sp_client stream_connection;
        UltraGridManager UG;
        wxArrayInt playList;
        Settings settings;

        DECLARE_EVENT_TABLE()
    friend class UltraGridManager;
};

#endif // CLIENT_GUIMAIN_H
