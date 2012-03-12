/***************************************************************
 * Name:      client_guiMain.cpp
 * Purpose:   Code for Application Frame
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#include "client_guiMain.h"
#include "CompressionSetting.h"
#include "ServerSelectionDialog.h"
#include "include/ClientDataIntPair.h"

#include <wx/msgdlg.h>
#include <string>
#include <iostream>
#include <stdexcept>

//(*InternalHeaders(client_guiFrame)
#include <wx/string.h>
#include <wx/intl.h>
//*)

wxCommandEvent defaultCommandEvent;

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
const long client_guiFrame::ID_GLCANVAS1 = wxNewId();
const long client_guiFrame::ID_BUTTON2 = wxNewId();
const long client_guiFrame::ID_TEXTCTRL1 = wxNewId();
const long client_guiFrame::ID_BUTTON3 = wxNewId();
const long client_guiFrame::ID_FR_LABEL = wxNewId();
const long client_guiFrame::ID_FR = wxNewId();
const long client_guiFrame::ID_SLIDER1 = wxNewId();
const long client_guiFrame::PlayButton = wxNewId();
const long client_guiFrame::ID_BUTTON1 = wxNewId();
const long client_guiFrame::idMenuQuit = wxNewId();
const long client_guiFrame::idServerSetting = wxNewId();
const long client_guiFrame::idCompressionSetting = wxNewId();
const long client_guiFrame::idMenuAbout = wxNewId();
const long client_guiFrame::ID_STATUSBAR1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(client_guiFrame,wxFrame)
    //(*EventTable(client_guiFrame)
    //*)
    EVT_COMMAND  (wxID_ANY, wxEVT_RECONF, client_guiFrame::Resize)
    EVT_COMMAND  (wxID_ANY, wxEVT_UPDATE_TIMER, client_guiFrame::UpdateTimer)
    EVT_COMMAND  (wxID_ANY, wxEVT_TOGGLE_FULLSCREEN, client_guiFrame::ToggleFullscreen)
    EVT_COMMAND  (wxID_ANY, wxEVT_TOGGLE_PAUSE, client_guiFrame::TogglePause)
    EVT_COMMAND  (wxID_ANY, wxEVT_SCROLLED, client_guiFrame::Scrolled)
    EVT_MOTION(client_guiFrame::MouseMotion)
    EVT_KEY_DOWN(client_guiFrame::KeyDown)
END_EVENT_TABLE()

client_guiFrame::client_guiFrame(wxWindow* parent,wxWindowID id) :
    UG(this),
    msgHandler(this)
{
    //(*Initialize(client_guiFrame)
    wxMenuItem* MenuItem2;
    wxMenuItem* MenuItem1;
    wxMenu* Menu1;
    wxMenu* Menu3;
    wxMenuItem* MenuItem3;
    wxMenuBar* MenuBar1;
    wxMenu* Menu2;

    Create(parent, wxID_ANY, _("FlashNET Player"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_FRAME_STYLE|wxWANTS_CHARS, _T("wxID_ANY"));
    FlexGridSizer1 = new CustomGridBagSizer(2, 1, 0, 0);
    FlexGridSizer1->AddGrowableCol(0);
    FlexGridSizer1->AddGrowableRow(0);
    int GLCanvasAttributes_1[] = {
    	WX_GL_RGBA,
    	WX_GL_DOUBLEBUFFER,
    	WX_GL_DEPTH_SIZE,      16,
    	WX_GL_STENCIL_SIZE,    0,
    	0, 0 };
    gl = new GLView(this, ID_GLCANVAS1, wxDefaultPosition, wxSize(487,234), 0, _T("ID_GLCANVAS1"), GLCanvasAttributes_1);
    FlexGridSizer1->Add(gl, 1, wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FlexGridSizer2 = new wxFlexGridSizer(2, 8, 0, 0);
    FlexGridSizer2->AddGrowableCol(5);
    Select = new wxButton(this, ID_BUTTON2, _("Select video"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
    FlexGridSizer2->Add(Select, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    fps = new wxTextCtrl(this, ID_TEXTCTRL1, _("FPS"), wxDefaultPosition, wxSize(45,25), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
    FlexGridSizer2->Add(fps, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FPSOk = new wxButton(this, ID_BUTTON3, _("OK"), wxDefaultPosition, wxSize(27,27), 0, wxDefaultValidator, _T("ID_BUTTON3"));
    FlexGridSizer2->Add(FPSOk, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FrameCountLabel = new wxStaticText(this, ID_FR_LABEL, _("FR"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_FR_LABEL"));
    FlexGridSizer2->Add(FrameCountLabel, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FrameCount = new wxSpinCtrl(this, ID_FR, _T("0"), wxDefaultPosition, wxSize(67,25), 0, 0, 2592000, 0, _T("ID_FR"));
    FrameCount->SetValue(_T("0"));
    FlexGridSizer2->Add(FrameCount, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Slider1 = new ProgressSlider(this, ID_SLIDER1, 0, 0, 100, wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_SLIDER1"));
    Slider1->SetMinSize(wxSize(100,-1));
    FlexGridSizer2->Add(Slider1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    StopBtn = new wxButton(this, PlayButton, _("Stop"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("PlayButton"));
    StopBtn->SetMaxSize(wxSize(-1,-1));
    FlexGridSizer2->Add(StopBtn, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Pause = new wxButton(this, ID_BUTTON1, _("Pause"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON1"));
    FlexGridSizer2->Add(Pause, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FlexGridSizer1->Add(FlexGridSizer2, 1, wxALL|wxEXPAND|wxFIXED_MINSIZE|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    SetSizer(FlexGridSizer1);
    MenuBar1 = new wxMenuBar();
    Menu1 = new wxMenu();
    MenuItem1 = new wxMenuItem(Menu1, idMenuQuit, _("Quit\tAlt-F4"), _("Quit the application"), wxITEM_NORMAL);
    Menu1->Append(MenuItem1);
    MenuBar1->Append(Menu1, _("&File"));
    Menu3 = new wxMenu();
    MenuItem3 = new wxMenuItem(Menu3, idServerSetting, _("Server"), _("Set server"), wxITEM_NORMAL);
    Menu3->Append(MenuItem3);
    MenuItem4 = new wxMenuItem(Menu3, idCompressionSetting, _("Compression"), _("Sets preferred compression"), wxITEM_NORMAL);
    Menu3->Append(MenuItem4);
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

    gl->Connect(wxEVT_PAINT,(wxObjectEventFunction)&client_guiFrame::OnGLCanvas1Paint,0,this);
    Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnSelectClick);
    Connect(ID_TEXTCTRL1,wxEVT_COMMAND_TEXT_UPDATED,(wxObjectEventFunction)&client_guiFrame::OnTextCtrl1Text);
    Connect(ID_BUTTON3,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnButton1Click2);
    Connect(ID_FR,wxEVT_COMMAND_SPINCTRL_UPDATED,(wxObjectEventFunction)&client_guiFrame::OnFrameCountChange);
    Connect(PlayButton,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnStopBtnClick);
    Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnPauseClick);
    Connect(idMenuQuit,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnQuit);
    Connect(idMenuAbout,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnAbout);
    //*)
    Connect(idServerSetting,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnServerSetting);
    Connect(idCompressionSetting,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnCompressSetting);
    gl->Connect(wxEVT_PAINT,(wxObjectEventFunction)&GLView::OnPaint,0,gl);
    /*int GLCanvasAttributes_1[] = {
    	WX_GL_RGBA,
    	WX_GL_DOUBLEBUFFER,
    	WX_GL_DEPTH_SIZE,      16,
    	WX_GL_STENCIL_SIZE,    0,
    	0, 0 };
    gl = new GLView(this, GLCanvasAttributes_1, FlexGridSizer1);
    //gl->PostInit();
    //gl->SetSize(wxSize(1000,1000));
    FlexGridSizer1->Insert(0, gl, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);*/
    //gl->SetSize(wxSize(100,100));

    /*GLCanvas1 = new wxGLCanvas(this, ID_GLCANVAS1, wxDefaultPosition, wxSize(66,69), 0, _T("ID_GLCANVAS1"), GLCanvasAttributes_1);
    FlexGridSizer1->Add(GLCanvas1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);*/

    FlexGridSizer1->Fit(this);
    FlexGridSizer1->SetSizeHints(this);

    /*SetSizer(FlexGridSizer1);
    SetAutoLayout(true);*/
    receiver = new UGReceiver(this, (const char *) "wxgl", gl);

    selectVideo = new VideoSelection(this, &settings);

    stream_connection.SetMsgHandler(&msgHandler);

    this->SetSize(600, 400);
    ChangeState(sInit);
/*
    wxAcceleratorEntry entries[14];
    entries[0].Set(wxACCEL_NORMAL, (int) WXK_SPACE, PlayButton);
    wxAcceleratorTable accel(1, entries);
    SetAcceleratorTable(accel);*/
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

void client_guiFrame::OnCompressSetting(wxCommandEvent& event)
{
    CompressionSetting dlg(this);
    wxString compression;
    wxString JPEGQuality;
    compression = wxString(settings.GetValue(std::string("compression"), std::string("none")).c_str(), wxConvUTF8);
    JPEGQuality = wxString(settings.GetValue(std::string("jpeg_qual"), std::string("80")).c_str(), wxConvUTF8);
    dlg.SetValue(compression);
    long qual;
    JPEGQuality.ToLong(&qual);
    dlg.SpinCtrl1->SetValue(qual);
    if ( dlg.ShowModal() == wxID_OK ) {
        //server = dlg.TextCtrl1->GetValue();
        //settings.SetValue("server", std::string(server.mb_str()));
        compression = dlg.Choice1->GetString(dlg.Choice1->GetSelection());
        settings.SetValue("compression", std::string(compression.mb_str()));
        JPEGQuality = wxString::Format(wxT("%i"), dlg.SpinCtrl1->GetValue());
        settings.SetValue("jpeg_qual", std::string(JPEGQuality.mb_str()));
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

void client_guiFrame::OnListBox1Select1(wxCommandEvent& event)
{
}

void client_guiFrame::OnButton1Click(wxCommandEvent& event)
{


}

void client_guiFrame::OnStopBtnClick(wxCommandEvent& event)
{
    if(state == sInit) {
        // nothing
    } else if(state == sReady) {
        Stop();
    } else {
        Stop();
    }
}

void client_guiFrame::PlaySelection()
{
    wxString item;
    wxString hostname;
    wxString path;
    wxCharBuffer buf;

    /*if(UG.StopRunning()) { // stopped
        return;
    }*/

    if(this->playList.empty()) {
        StatusBar1->PushStatusText(wxT("No video selected!"));
        //wxMessageBox(_("No video selected"), _("Unable to play"));
        return;
    }

    try {
        item = this->playList[0].URL;
        item = item.Mid(wxString(L"rtp://").Len());
        hostname = item.BeforeFirst('/');
        path = item.AfterFirst('/');
        wxString video_format = this->playList[0].format;

        double fps;
        if(!this->fps->GetValue().ToDouble(&fps)) {
             /* error! */
        }

        //this->playList.pop_back();

        this->stream_connection.connect_to(std::string(hostname.mb_str()), 5100);

        wxString msgstr;
        struct message msg;
        struct response resp;

        msgstr = L"";
        msgstr << wxT("SET_PARAMETER format ") << video_format;
        buf = msgstr.mb_str();
        msg.msg = buf.data();
        msg.len = msgstr.Len();

        this->stream_connection.send(&msg, &resp);

        if(resp.code != 200) {
            wxString msg;
            msg << wxT("Unable to set video format: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
            throw std::runtime_error(std::string(msg.mb_str()));
        }

        msgstr = L"";
        msgstr << wxT("SET_PARAMETER compression ") << wxString(settings.GetValue(std::string("compression"), std::string("none")).c_str(), wxConvUTF8) << wxT(" ") <<
                wxString(settings.GetValue(std::string("jpeg_qual"), std::string("80")).c_str(), wxConvUTF8);
        buf = msgstr.mb_str();
        msg.msg = buf.data();
        msg.len = msgstr.Len();

        this->stream_connection.send(&msg, &resp);

        if(resp.code != 200) {
            wxString msg;
            msg << wxT("Unable to set compression: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
            throw std::runtime_error(std::string(msg.mb_str()));
        }

        msgstr = L"";
        msgstr << wxT("SETUP /") << path;
        buf = msgstr.mb_str();
        msg.msg = buf.data();
        msg.len = msgstr.Len();

        // TODO: handle TMOUT
        this->stream_connection.send(&msg, &resp);

        if(resp.code != 201) {
            wxString msg;
            msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
            wxMessageBox(msg, _("Error connecting to server"));
            return;
        }

        msgstr = L"";
        msgstr << wxT("SET_PARAMETER fps ") << wxString::Format(wxT("%2.2f"),fps);
        std::cerr << std::string(msgstr.mb_str());
        buf = msgstr.mb_str();
        msg.msg = buf.data();
        msg.len = msgstr.Len();

        this->stream_connection.send(&msg, &resp);

        if(resp.code != 200) {
            wxString msg;
            msg << wxT("Unable to set fps: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
            throw std::runtime_error(std::string(msg.mb_str()));
        }


        gl->Receive(true);

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

        //UG.newWindow();
        StatusBar1->PushStatusText(item);
        ChangeState(sPlaying);
        total_frames = this->playList[0].total_frames;

    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error downloading media file"));
        this->stream_connection.disconnect();
    }
}

void client_guiFrame::Stop()
{
    {
        wxString msgstr;
        struct message msg;
        struct response resp;

        msg.msg = "TEARDOWN";
        msg.len = strlen(msg.msg);

        // TODO: handle TMOUT
        this->stream_connection.send(&msg, &resp);

        if(resp.code != 200) {
            wxString msg;
            msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
            wxMessageBox(msg, _("Error stopping media"));
            return;
        }

        StatusBar1->PushStatusText(wxT("stopped"));
    }
    this->stream_connection.disconnect();
    ChangeState(sInit);
    DoUpdateCounters(0);
    //Refresh();
    gl->LoadSplashScreen();
    gl->Receive(false);
    total_frames = 0;
}

void client_guiFrame::NotifyWindowClosed()
{
    wxString msgstr;
    struct message msg;
    struct response resp;

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

void client_guiFrame::Resize(wxCommandEvent& event)
{
    ClientDataIntPair *pair = dynamic_cast<ClientDataIntPair *>(event.GetClientObject());
    //this->Layout();

    this->SetSize(GetSize().x + pair->first(), GetSize().y + pair->second());
    //this->Fit();
}

void client_guiFrame::OnSelectClick(wxCommandEvent& event)
{

    if(selectVideo->ShowModal() == wxOK) {
    }
    this->playList.Empty();
    this->playList = selectVideo->GetSelectedVideo();
    if(!this->playList.IsEmpty())
        fps->SetValue(wxString::Format(wxT("%2.2f"), this->playList[0].fps));
    /*selectVideo->ListBox1->GetSelections(playListIndices);

    for (int i = 0; i < playListIndices.Count(); ++i) {
        this->playList.Insert(selectVideo->ListBox1->GetString(playListIndices[i]), i);
    }*/
}

void client_guiFrame::DataReceived()
{
    stream_connection.ProcessIncomingData();
}

void client_guiFrame::DoDisconnect()
{
    StatusBar1->PushStatusText(wxT("interrupted by server"));
    ChangeState(sInit);
    stream_connection.disconnect();
}

void client_guiFrame::OnTextCtrl1Text(wxCommandEvent& event)
{
}

void client_guiFrame::DoUpdateCounters(int val)
{
    if(!this->playList.IsEmpty()) {
        Slider1->SetValue((double)  val / (this->total_frames - 1) * 100.0);

        if(Slider1->GetValue() == 100) {
            DoPause();
        }

        FrameCount->SetValue(val);
    } else {
        Slider1->SetValue(0);
        FrameCount->SetValue(0);
    }
}

void client_guiFrame::UpdateTimer(wxCommandEvent&)
{
    if(state != sInit)
        DoUpdateCounters(gl->GetFrameSeq());
}

void client_guiFrame::JumpToFrame(int frame)
{
    wxString msgstr;
    char msgbuf[40];
    struct message msg;
    struct response resp;

    if(frame < 0 || frame >= this->total_frames)
        return;

    if(state == sPlaying) {
        snprintf(msgbuf, 40, "PLAY %d", frame);
    } else if (state == sReady) {
        snprintf(msgbuf, 40, "PAUSE %d", frame);
    } else {
        return;
    }

    msg.msg = msgbuf;
    msg.len = strlen(msg.msg);

    this->stream_connection.send(&msg, &resp);

    if(resp.code != 200) {
        wxString msg;
        msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
        wxMessageBox(msg, _("Error setting position"));
    }
}

void client_guiFrame::Scrolled(wxCommandEvent&)
{
    if(this->playList.IsEmpty())
        return;
    int new_pos = round((double) Slider1->GetValue() / 100.0 * (this->total_frames - 1));

    JumpToFrame(new_pos);
}

void client_guiFrame::OnPauseClick(wxCommandEvent& event)
{
    //assert(event.GetExtraLong() != MOUSE_CLICKED_MAGIC);
    if(state == sInit && event.GetExtraLong() != MOUSE_CLICKED_MAGIC) {
        PlaySelection();
    } else if(state == sPlaying) {
        DoPause();
    } else if(state == sReady) {
        Resume();
    }
}

void client_guiFrame::DoPause()
{
    if(state != sPlaying)
        return;

    try {
        wxString msgstr;
        struct message msg;
        struct response resp;

        msg.msg = "PAUSE";
        msg.len = strlen(msg.msg);

        // TODO: handle TMOUT
        this->stream_connection.send(&msg, &resp);

        if(resp.code != 200) {
            wxString msg;
            msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
            wxMessageBox(msg, _("Error pausing media"));
        } else {
            ChangeState(sReady);
        }
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error downloading media file"));
    }
}

void client_guiFrame::Resume()
{
    wxString msgstr;
    struct message msg;
    struct response resp;

    msg.msg = "PLAY";
    msg.len = strlen(msg.msg);

    try {
        // TODO: handle TMOUT
        this->stream_connection.send(&msg, &resp);

        if(resp.code != 200) {
            wxString msg;
            msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
            wxMessageBox(msg, _("Error resuming media"));
        } else {
            ChangeState(sPlaying);
            Pause->SetLabel(wxT("Pause"));
        }
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error downloading media file"));
    }
}

void client_guiFrame::ChangeState(enum playerState newState)
{
    state = newState;
    switch(newState) {
        case sInit:
            Pause->SetLabel(wxT("Play"));
            break;
        case sReady:
            Pause->SetLabel(wxT("Play"));
            break;
        case sPlaying:
            Pause->SetLabel(wxT("Pause"));
            break;
    }
}

void client_guiFrame::OnButton1Click2(wxCommandEvent& event)
{
    wxString msgstr;
    struct message msg;
    struct response resp;

    if(!this->stream_connection.isConnected())
        return;

    try {
        double fps;
        if(!this->fps->GetValue().ToDouble(&fps)) {
             /* error! */
        }


        msgstr << wxT("SET_PARAMETER fps ") << wxString::Format(wxT("%2.2f"),fps);
        wxCharBuffer buf = msgstr.mb_str();
        msg.msg = buf.data();
        msg.len = msgstr.Len();

        this->stream_connection.send(&msg, &resp);

        if(resp.code != 200) {
            wxString msg;
            msg << wxT("Unable to set fps: ") << resp.code << L" " << wxString::FromUTF8((resp.msg));
            throw std::runtime_error(std::string(msg.mb_str()));
        }
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error setting fps"));
    }
}

void client_guiFrame::OnGLCanvas1Paint(wxPaintEvent& event)
{
}

void client_guiFrame::ToggleFullscreen(wxCommandEvent& evt = defaultCommandEvent)
{
    wxWindow *saved;
    if(IsFullScreen()) {
        //wxGBSpan span(1,1);
        //assert(FlexGridSizer1->SetItemSpan((size_t) 0, span));

        //FlexGridSizer1->Detach((size_t) 1);
        //FlexGridSizer1->RecalcSizes();
        FlexGridSizer2->Show(true);
        ShowFullScreen(false, wxFULLSCREEN_ALL);
    } else {
        //wxGBSpan span(2,2);
        //FlexGridSizer2->Show(false);
        //FlexGridSizer1->SetItemSpan((size_t) 1, span);
        //wxGBPosition pos;
        //pos = FlexGridSizer1->GetItemPosition((size_t) 1);
        //span = FlexGridSizer1->GetItemSpan((size_t) 1);
        //std::cerr << span.GetColspan() << span.GetRowspan() << std::endl;
        //FlexGridSizer1->Detach((size_t) 1);
        /*span = wxGBSpan(0,0);
        FlexGridSizer1->SetItemSpan((size_t) 2, span);*/
        FlexGridSizer2->Show(false);
        ShowFullScreen(true, wxFULLSCREEN_ALL);
    }
}

void client_guiFrame::TogglePause(wxCommandEvent& evt)
{
    OnPauseClick(evt);
}

void client_guiFrame::MouseMotion(wxMouseEvent& evt)
{
    if(IsFullScreen()  && !FlexGridSizer1->IsShown(FlexGridSizer2) && evt.GetY() > (GetSize().y - 5)) {
        //std::cerr << "." << evt.GetY();
        FlexGridSizer2->Show(true);
        FlexGridSizer1->Layout();
        this->Fit();
    } else if(IsFullScreen() && FlexGridSizer1->IsShown(FlexGridSizer2)  && evt.GetY() < (GetSize().y - 20)) {
        //std::cerr << "!" << evt.GetY();
        FlexGridSizer2->Show(false);
        FlexGridSizer1->Layout();
        this->Fit();
    }
}

void client_guiFrame::OnFrameCountChange(wxSpinEvent& event)
{
    JumpToFrame(event.GetPosition());
}

void client_guiFrame::KeyDown(wxKeyEvent& evt)
{
    switch(evt.GetUnicodeKey()) {
        case WXK_LEFT:
        case WXK_RIGHT:
        case WXK_UP:
        case WXK_DOWN:
        case WXK_SPACE:
             if(state == sPlaying) {
                DoPause();
             } else if(state == sReady) {
                Resume();
             }
            break;
        case WXK_ESCAPE:
            if(IsFullScreen()) {
                ToggleFullscreen();
            }
            break;
        case 'F':
            ToggleFullscreen();
            break;
        case '\r':
            if(evt.GetModifiers() == wxMOD_ALT) {
                ToggleFullscreen();
            }
            break;
        case 'K':
            if(state == sPlaying)
                DoPause();
            break;
        case 'J':
            JumpToFrame(gl->GetFrameSeq() - 1);
            break;
        case 'L':
            JumpToFrame(gl->GetFrameSeq() + 1);
            break;

        default:
            std::cerr << evt.GetUnicodeKey() << " " << (int) '\r' << std::endl;
    }
}

