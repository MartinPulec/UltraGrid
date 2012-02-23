/***************************************************************
 * Name:      client_guiMain.cpp
 * Purpose:   Code for Application Frame
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#include "client_guiMain.h"
#include "ServerSelectionDialog.h"
#include <wx/msgdlg.h>
#include <string>
#include <iostream>

//(*InternalHeaders(client_guiFrame)
#include <wx/string.h>
#include <wx/intl.h>
//*)

#include <wx/tokenzr.h>

//helper functions
enum wxbuildinfoformat {
    short_f, long_f };

wxString wxbuildinfo(wxbuildinfoformat format)
{
    wxString wxbuild(wxVERSION_STRING);

    if (format == long_f )
    {
#if defined(__WXMSW__)
        wxbuild << _T("-Windows");
#elif defined(__UNIX__)
        wxbuild << _T("-Linux");
#endif

#if wxUSE_UNICODE
        wxbuild << _T("-Unicode build");
#else
        wxbuild << _T("-ANSI build");
#endif // wxUSE_UNICODE
    }

    return wxbuild;
}

//(*IdInit(client_guiFrame)
const long client_guiFrame::ID_LISTBOX1 = wxNewId();
const long client_guiFrame::ID_BUTTON1 = wxNewId();
const long client_guiFrame::PlayButton = wxNewId();
const long client_guiFrame::idMenuQuit = wxNewId();
const long client_guiFrame::idServerSetting = wxNewId();
const long client_guiFrame::idMenuAbout = wxNewId();
const long client_guiFrame::ID_STATUSBAR1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(client_guiFrame,wxFrame)
    //(*EventTable(client_guiFrame)
    //*)
END_EVENT_TABLE()

client_guiFrame::client_guiFrame(wxWindow* parent,wxWindowID id) :
    UG(this)
{
    //(*Initialize(client_guiFrame)
    wxMenuItem* MenuItem2;
    wxMenuItem* MenuItem1;
    wxFlexGridSizer* FlexGridSizer1;
    wxMenu* Menu1;
    wxMenu* Menu3;
    wxMenuItem* MenuItem3;
    wxBoxSizer* BoxSizer1;
    wxMenuBar* MenuBar1;
    wxMenu* Menu2;

    Create(parent, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_FRAME_STYLE, _T("wxID_ANY"));
    FlexGridSizer1 = new wxFlexGridSizer(2, 1, 0, 0);
    FlexGridSizer1->AddGrowableCol(0);
    FlexGridSizer1->AddGrowableRow(0);
    ListBox1 = new wxListBox(this, ID_LISTBOX1, wxDefaultPosition, wxSize(445,181), 0, 0, 0, wxDefaultValidator, _T("ID_LISTBOX1"));
    FlexGridSizer1->Add(ListBox1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    BoxSizer1 = new wxBoxSizer(wxHORIZONTAL);
    Button1 = new wxButton(this, ID_BUTTON1, _("Get list"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON1"));
    BoxSizer1->Add(Button1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    BoxSizer1->Add(-1,-1,1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Play = new wxButton(this, PlayButton, _("Start stream"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("PlayButton"));
    BoxSizer1->Add(Play, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FlexGridSizer1->Add(BoxSizer1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    SetSizer(FlexGridSizer1);
    MenuBar1 = new wxMenuBar();
    Menu1 = new wxMenu();
    MenuItem1 = new wxMenuItem(Menu1, idMenuQuit, _("Quit\tAlt-F4"), _("Quit the application"), wxITEM_NORMAL);
    Menu1->Append(MenuItem1);
    MenuBar1->Append(Menu1, _("&File"));
    Menu3 = new wxMenu();
    MenuItem3 = new wxMenuItem(Menu3, idServerSetting, _("Server"), _("Set server"), wxITEM_NORMAL);
    Menu3->Append(MenuItem3);
    MenuBar1->Append(Menu3, _("&Settings"));
    Menu2 = new wxMenu();
    MenuItem2 = new wxMenuItem(Menu2, idMenuAbout, _("About\tF1"), _("Show info about this application"), wxITEM_NORMAL);
    Menu2->Append(MenuItem2);
    MenuBar1->Append(Menu2, _("Help"));
    SetMenuBar(MenuBar1);
    StatusBar1 = new wxStatusBar(this, ID_STATUSBAR1, 0, _T("ID_STATUSBAR1"));
    int __wxStatusBarWidths_1[1] = { -1 };
    int __wxStatusBarStyles_1[1] = { wxSB_NORMAL };
    StatusBar1->SetFieldsCount(1,__wxStatusBarWidths_1);
    StatusBar1->SetStatusStyles(1,__wxStatusBarStyles_1);
    SetStatusBar(StatusBar1);
    FlexGridSizer1->Fit(this);
    FlexGridSizer1->SetSizeHints(this);

    Connect(ID_LISTBOX1,wxEVT_COMMAND_LISTBOX_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnListBox1Select1);
    Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnButton1Click);
    Connect(PlayButton,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnPlayClick);
    Connect(idMenuQuit,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnQuit);
    Connect(idMenuAbout,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnAbout);
    //*)
    Connect(idServerSetting,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnServerSetting);
}

client_guiFrame::~client_guiFrame()
{
    //(*Destroy(client_guiFrame)
    //*)
}

void client_guiFrame::OnServerSetting(wxCommandEvent& event)
{
    ServerSelectionDialog dlg(this);
    wxString server;
    server = wxString(settings.GetValue(std::string("server")).c_str(), wxConvUTF8);
    dlg.TextCtrl1->SetValue(server);
    if ( dlg.ShowModal() == wxID_OK ) {
        server = dlg.TextCtrl1->GetValue();
        settings.SetValue("server", std::string(server.mb_str()));
    } else {
        //else: dialog was cancelled or some another button pressed
    }
}

void client_guiFrame::OnQuit(wxCommandEvent& event)
{
    Close();
}

void client_guiFrame::OnAbout(wxCommandEvent& event)
{
    wxString msg = wxbuildinfo(long_f);
    wxMessageBox(msg, _("Welcome to..."));
}

void client_guiFrame::OnListBox1Select(wxCommandEvent& event)
{
}

void client_guiFrame::OnListBox1Select1(wxCommandEvent& event)
{
}

void client_guiFrame::OnButton1Click(wxCommandEvent& event)
{
    sp_client connection;

    try {
        connection.connect_to(settings.GetValue("server"), 5100);
        struct message get_req;
        struct response resp;
        get_req.msg = "GET index.lst\r\n";
        get_req.len = strlen(get_req.msg);
        connection.send(&get_req, &resp);
        if(resp.code != 200) {
            wxString msg;
            msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
            wxMessageBox(msg, _("Error downloading media file"));
        }


        ListBox1->Clear();

        wxString data = wxString::FromUTF8(resp.body, resp.body_len);
        wxStringTokenizer tkz(data, wxT("\r\n"));
        while ( tkz.HasMoreTokens() )
        {
            wxString token = tkz.GetNextToken();
            ListBox1->InsertItems(1u, &token, 0u);
        }
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error downloading media file"));
    }

}

void client_guiFrame::OnPlayClick(wxCommandEvent& event)
{
    ListBox1->GetSelections(this->playList);
    PlaySelection();
}

void client_guiFrame::SetPlayButtonStop(bool stop)
{
    if(stop) {
        Play->SetLabel(wxT("Stop stream"));
    } else {
        Play->SetLabel(wxT("Start stream"));
    }
}

void client_guiFrame::PlaySelection()
{
    wxString item;
    wxString hostname;
    wxString path;

    if(UG.StopRunning()) { // stopped
        return;
    }

    if(this->playList.empty())
        return;

    try {
        item = ListBox1->GetString(this->playList.front());
        item = item.Mid(wxString(L"rtp://").Len());
        hostname = item.BeforeFirst('/');
        path = item.AfterFirst('/');

        this->playList.pop_back();

        this->stream_connection.connect_to(std::string(hostname.mb_str()), 5100);

        {
            wxString msgstr;
            struct message msg;
            struct response resp;

            msgstr << wxT("SETUP /") << path;
            wxCharBuffer buf = msgstr.mb_str();
            msg.msg = buf.data();
            msg.len = msgstr.Len();

            // TODO: handle TMOUT
            this->stream_connection.send(&msg, &resp);

            if(resp.code != 201) {
                wxString msg;
                msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
                wxMessageBox(msg, _("Error downloading media file"));
                return;
            }

            msgstr = L"";
            msgstr << wxT("PLAY");
            buf = msgstr.mb_str();
            msg.msg = buf.data();
            msg.len = msgstr.Len();

            // TODO: handle TMOUT
            this->stream_connection.send(&msg, &resp);

            if(resp.code != 200) {
                wxString msg;
                msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
                wxMessageBox(msg, _("Error downloading media file"));
                return;
            }

            UG.newWindow();
            SetPlayButtonStop(true);
        }

    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error downloading media file"));
    }
}

void client_guiFrame::NotifyWindowClosed()
{
    wxString msgstr;
    struct message msg;
    struct response resp;

    SetPlayButtonStop(false);

    try {
        msgstr << wxT("TEARDOWN");
        wxCharBuffer buf = msgstr.mb_str();
        msg.msg = buf.data();
        msg.len = msgstr.Len();

        // TODO: handle TMOUT
        this->stream_connection.send(&msg, &resp);

        if(resp.code != 200) {
            wxString msg;
            msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
            wxMessageBox(msg, _("Error closing connection"));
            return;
        }

        this->stream_connection.disconnect();
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error downloading media file"));
    }
}
