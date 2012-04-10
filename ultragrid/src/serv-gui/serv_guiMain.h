/***************************************************************
 * Name:      serv_guiMain.h
 * Purpose:   Defines Application Frame
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#ifndef SERV_GUIMAIN_H
#define SERV_GUIMAIN_H

#include <wx/grid.h>
//(*Headers(serv_guiFrame)
#include <wx/checklst.h>
#include <wx/sizer.h>
#include <wx/button.h>
#include <wx/menu.h>
#include <wx/dirdlg.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/stattext.h>
//*)

#include "../client-gui/include/VideoEntry.h"

class serv_guiFrame: public wxFrame
{
    public:

        serv_guiFrame(wxWindow* parent,wxWindowID id = -1);
        virtual ~serv_guiFrame();

    private:

        //(*Handlers(serv_guiFrame)
        void OnQuit(wxCommandEvent& event);
        void OnAbout(wxCommandEvent& event);
        void OnCheckListBox1Toggled(wxCommandEvent& event);
        void OnButton1Click(wxCommandEvent& event);
        void OnButton2Click(wxCommandEvent& event);
        void OnButton3Click(wxCommandEvent& event);
        void OnButton4Click(wxCommandEvent& event);
        //*)

        void LoadSettings(void);

        void ScanDirectory(wxString path, VideoEntry &entry);

        wxString filename;
        ArrayOfVideoEntries videos;

        //(*Identifiers(serv_guiFrame)
        static const long ID_STATICTEXT1;
        static const long ID_CHECKLISTBOX1;
        static const long ID_BUTTON1;
        static const long ID_BUTTON2;
        static const long ID_Edit;
        static const long ID_BUTTON3;
        static const long idMenuQuit;
        static const long idMenuAbout;
        static const long ID_STATUSBAR1;
        //*)

        //(*Declarations(serv_guiFrame)
        wxStatusBar* StatusBar1;
        wxButton* Button1;
        wxButton* Button2;
        wxButton* Button3;
        wxDirDialog* PathSelection;
        wxStaticText* StaticText1;
        wxCheckListBox* CheckListBox1;
        wxButton* Edit;
        //*)

        DECLARE_EVENT_TABLE()
};

#endif // SERV_GUIMAIN_H
