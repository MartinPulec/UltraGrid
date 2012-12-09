/***************************************************************
 * Name:      client_guiMain.cpp
 * Purpose:   Code for Application Frame
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#include "client_guiMain.h"
#include "About.h"
#include "CompressionSetting.h"
#include "ServerSelectionDialog.h"
#include "OtherSettingsDialog.h"
#include "KeyBindingsHelp.h"
#include "include/ClientDataIntPair.h"
#include "include/ClientDataCStr.h"
#include "include/ClientDataHWDisplay.h"
#include "ClientDataWeakGenericPtr.h"
#include "include/Utils.h"

#include "video_display.h"
#include "audio/audio_playback.h"

#include <wx/msgdlg.h>
#include <string>
#include <iostream>
#include <stdexcept>

//(*InternalHeaders(client_guiFrame)
#include <wx/string.h>
#include <wx/intl.h>
//*)

wxCommandEvent defaultCommandEvent;

using namespace std;

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
const long client_guiFrame::ID_ToggleLoop = wxNewId();
const long client_guiFrame::ID_SLIDER1 = wxNewId();
const long client_guiFrame::ID_Backward = wxNewId();
const long client_guiFrame::ID_SPEED_STR = wxNewId();
const long client_guiFrame::ID_Forward = wxNewId();
const long client_guiFrame::ID_Slower = wxNewId();
const long client_guiFrame::ID_Quicker = wxNewId();
const long client_guiFrame::PlayButton = wxNewId();
const long client_guiFrame::ID_BUTTON1 = wxNewId();
const long client_guiFrame::ID_HD = wxNewId();
const long client_guiFrame::ID_J2K_QUALITY_LABEL = wxNewId();
const long client_guiFrame::ID_J2KBitrateVal = wxNewId();
const long client_guiFrame::ID_J2K_QUALITY_SLIDER = wxNewId();
const long client_guiFrame::idMenuQuit = wxNewId();
const long client_guiFrame::idServerSetting = wxNewId();
const long client_guiFrame::idCompressionSetting = wxNewId();
const long client_guiFrame::idOtherSettings = wxNewId();
const long client_guiFrame::idMenuAbout = wxNewId();
const long client_guiFrame::idKeyBindings = wxNewId();
const long client_guiFrame::ID_STATUSBAR1 = wxNewId();
//*)

DEFINE_EVENT_TYPE(wxEVT_DISCONNECT)

BEGIN_EVENT_TABLE(client_guiFrame,wxFrame)
    //(*EventTable(client_guiFrame)
    //*)
    EVT_COMMAND  (wxID_ANY, wxEVT_RECONF, client_guiFrame::Resize)
    EVT_COMMAND  (wxID_ANY, wxEVT_DISCONNECT, client_guiFrame::PushDisconnect)
    EVT_COMMAND  (wxID_ANY, wxEVT_TOGGLE_FULLSCREEN, client_guiFrame::ToggleFullscreen)
    EVT_COMMAND  (wxID_ANY, wxEVT_TOGGLE_PAUSE, client_guiFrame::TogglePause)
    EVT_COMMAND  (wxID_ANY, wxEVT_SCROLLED, client_guiFrame::Scrolled)
    EVT_MOTION(client_guiFrame::Mouse)
    EVT_LEFT_UP(client_guiFrame::Mouse)
    EVT_KEY_DOWN(client_guiFrame::KeyDown)
    EVT_MOUSEWHEEL(client_guiFrame::Wheel)

    EVT_TOGGLEBUTTON(ID_ToggleLoop, client_guiFrame::OnButton1Click3)
END_EVENT_TABLE()

client_guiFrame::client_guiFrame(wxWindow* parent,wxWindowID id) :
    msgHandler(this)
{
    //(*Initialize(client_guiFrame)
    wxMenuItem* MenuItem2;
    wxMenuItem* MenuItem1;
    wxMenu* Menu1;
    wxMenu* Menu3;
    wxMenuItem* MenuItem3;
    wxFlexGridSizer* FlexGridSizer3;
    wxMenuItem* MenuItem5;
    wxMenuBar* MenuBar1;
    wxMenu* Menu2;

    Create(parent, wxID_ANY, _("FlashNET Player"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_FRAME_STYLE|wxWANTS_CHARS, _T("wxID_ANY"));
    FlexGridSizer1 = new CustomGridBagSizer(3, 1, 0, 0);
    FlexGridSizer1->AddGrowableCol(0);
    FlexGridSizer1->AddGrowableRow(0);
    int GLCanvasAttributes_1[] = {
    	WX_GL_RGBA,
    	WX_GL_DOUBLEBUFFER,
    	WX_GL_DEPTH_SIZE,      16,
    	WX_GL_STENCIL_SIZE,    0,
    	0, 0 };
    gl = new GLView(this, ID_GLCANVAS1, wxDefaultPosition, wxSize(767,391), 0, _T("ID_GLCANVAS1"), GLCanvasAttributes_1);
    FlexGridSizer1->Add(gl, 1, wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FlexGridSizer2 = new wxFlexGridSizer(1, 14, 0, 0);
    FlexGridSizer2->AddGrowableCol(6);
    Select = new wxButton(this, ID_BUTTON2, _("⏏"), wxDefaultPosition, wxSize(60,27), 0, wxDefaultValidator, _T("ID_BUTTON2"));
    FlexGridSizer2->Add(Select, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    fps = new wxTextCtrl(this, ID_TEXTCTRL1, _("FPS"), wxDefaultPosition, wxSize(45,25), 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
    FlexGridSizer2->Add(fps, 1, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FPSOk = new wxButton(this, ID_BUTTON3, _("OK"), wxDefaultPosition, wxSize(36,27), 0, wxDefaultValidator, _T("ID_BUTTON3"));
    FlexGridSizer2->Add(FPSOk, 1, wxTOP|wxBOTTOM|wxRIGHT|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FrameCountLabel = new wxStaticText(this, ID_FR_LABEL, _("FR"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_FR_LABEL"));
    FlexGridSizer2->Add(FrameCountLabel, 1, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FrameCount = new wxSpinCtrl(this, ID_FR, _T("0"), wxDefaultPosition, wxSize(67,25), 0, 0, 2592000, 0, _T("ID_FR"));
    FrameCount->SetValue(_T("0"));
    FlexGridSizer2->Add(FrameCount, 1, wxTOP|wxBOTTOM|wxRIGHT|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    ToggleLoop = new wxToggleButton(this, ID_ToggleLoop, _("Loop"), wxDefaultPosition, wxSize(60,-1), 0, wxDefaultValidator, _T("ID_ToggleLoop"));
    FlexGridSizer2->Add(ToggleLoop, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Slider1 = new ProgressSlider(this, ID_SLIDER1, 0, 0, 100, wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_SLIDER1"));
    Slider1->SetMinSize(wxSize(100,-1));
    FlexGridSizer2->Add(Slider1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Backward = new wxButton(this, ID_Backward, _("◀"), wxDefaultPosition, wxSize(27,27), 0, wxDefaultValidator, _T("ID_Backward"));
    FlexGridSizer2->Add(Backward, 1, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    SpeedStr = new wxStaticText(this, ID_SPEED_STR, _("SPD"), wxDefaultPosition, wxSize(41,15), 0, _T("ID_SPEED_STR"));
    FlexGridSizer2->Add(SpeedStr, 1, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Forward = new wxButton(this, ID_Forward, _("▶"), wxDefaultPosition, wxSize(27,27), 0, wxDefaultValidator, _T("ID_Forward"));
    FlexGridSizer2->Add(Forward, 1, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Slower = new wxButton(this, ID_Slower, _("▼"), wxDefaultPosition, wxSize(27,27), 0, wxDefaultValidator, _T("ID_Slower"));
    FlexGridSizer2->Add(Slower, 1, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Quicker = new wxButton(this, ID_Quicker, _("▲"), wxDefaultPosition, wxSize(27,27), 0, wxDefaultValidator, _T("ID_Quicker"));
    FlexGridSizer2->Add(Quicker, 1, wxTOP|wxBOTTOM|wxRIGHT|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    StopBtn = new wxButton(this, PlayButton, _("◼"), wxDefaultPosition, wxSize(60,27), 0, wxDefaultValidator, _T("PlayButton"));
    StopBtn->SetMaxSize(wxSize(-1,-1));
    FlexGridSizer2->Add(StopBtn, 1, wxTOP|wxBOTTOM|wxLEFT|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Pause = new wxButton(this, ID_BUTTON1, _("▶"), wxDefaultPosition, wxSize(60,27), 0, wxDefaultValidator, _T("ID_BUTTON1"));
    FlexGridSizer2->Add(Pause, 1, wxTOP|wxBOTTOM|wxRIGHT|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FlexGridSizer1->Add(FlexGridSizer2, 1, wxALL|wxEXPAND|wxFIXED_MINSIZE|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FlexGridSizer3 = new wxFlexGridSizer(1, 4, 0, 0);
    FlexGridSizer3->AddGrowableCol(3);
    FlexGridSizer3->AddGrowableRow(0);
    HD = new wxToggleButton(this, ID_HD, _("HD"), wxDefaultPosition, wxSize(43,29), 0, wxDefaultValidator, _T("ID_HD"));
    FlexGridSizer3->Add(HD, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    J2KQualityLabel = new wxStaticText(this, ID_J2K_QUALITY_LABEL, _("J2K bitrate:"), wxDefaultPosition, wxSize(81,17), 0, _T("ID_J2K_QUALITY_LABEL"));
    FlexGridSizer3->Add(J2KQualityLabel, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    J2KBitrateVal = new J2KBitrate(this, ID_J2KBitrateVal, _("N/A"), wxDefaultPosition, wxSize(54,17), 0, _T("ID_J2KBitrateVal"));
    FlexGridSizer3->Add(J2KBitrateVal, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    J2KQualitySlider = new wxSlider(this, ID_J2K_QUALITY_SLIDER, 1000, 0, 1000, wxDefaultPosition, wxSize(606,29), 0, wxDefaultValidator, _T("ID_J2K_QUALITY_SLIDER"));
    J2KQualitySlider->SetMaxSize(wxSize(-1,-1));
    FlexGridSizer3->Add(J2KQualitySlider, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FlexGridSizer1->Add(FlexGridSizer3, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
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
    MenuItem6 = new wxMenuItem(Menu3, idOtherSettings, _("Other Settings"), wxEmptyString, wxITEM_NORMAL);
    Menu3->Append(MenuItem6);
    MenuBar1->Append(Menu3, _("&Settings"));
    Menu2 = new wxMenu();
    MenuItem2 = new wxMenuItem(Menu2, idMenuAbout, _("About\tF1"), _("Show info about this application"), wxITEM_NORMAL);
    Menu2->Append(MenuItem2);
    MenuItem5 = new wxMenuItem(Menu2, idKeyBindings, _("Key Bindings"), _("Show list of keybindings"), wxITEM_NORMAL);
    Menu2->Append(MenuItem5);
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
    Connect(ID_ToggleLoop,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnButton1Click3);
    Connect(ID_Backward,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnBackwardSlowClick);
    Connect(ID_Forward,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnForwardSlowClick);
    Connect(ID_Slower,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnBackwardFastClick);
    Connect(ID_Quicker,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnForwardFastClick);
    Connect(PlayButton,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnStopBtnClick);
    Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnPauseClick);
    Connect(ID_HD,wxEVT_COMMAND_TOGGLEBUTTON_CLICKED,(wxObjectEventFunction)&client_guiFrame::OnHDToggle);
    Connect(idMenuQuit,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnQuit);
    Connect(idMenuAbout,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnAbout);
    //*)
    Connect(idServerSetting,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnServerSetting);
    Connect(idCompressionSetting,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnCompressSetting);
    Connect(idKeyBindings,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnKeyBindingsHelp);
    Connect(idOtherSettings,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&client_guiFrame::OnOtherSettings);

    wxEventType sliderEvent;
#ifdef __WXMAC__
    sliderEvent = wxEVT_SCROLL_THUMBRELEASE;
#else
    sliderEvent = wxEVT_SCROLL_CHANGED;
#endif

    J2KQualitySlider->Connect(sliderEvent,
        (wxObjectEventFunction)&Player::QualityChanged, 0, &player);
    J2KQualitySlider->Connect(sliderEvent,
        (wxObjectEventFunction)&J2KBitrate::QualityChanged, 0, J2KBitrateVal);
    fps->Connect(wxEVT_COMMAND_TEXT_UPDATED,
        (wxObjectEventFunction)&J2KBitrate::FPSChanged, 0, J2KBitrateVal);
    J2KBitrateVal->Update();

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

    selectVideo = new VideoSelection(this, &settings);

    player.SetMsgHandler(&msgHandler);

    //this->SetSize(600, 400);
/*
    wxAcceleratorEntry entries[14];
    entries[0].Set(wxACCEL_NORMAL, (int) WXK_SPACE, PlayButton);
    wxAcceleratorTable accel(1, entries);
    SetAcceleratorTable(accel);*/
    player.Init(gl, this, &settings);

    SpeedStr->SetLabel(Utils::FromCDouble(player.GetSpeed(), 2));
}

client_guiFrame::~client_guiFrame()
{
    //dtor
    //(*Destroy(client_guiFrame)
    //*)
    delete selectVideo;
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

void client_guiFrame::OnOtherSettings(wxCommandEvent& event)
{
    OtherSettingsDialog dlg(this);
    string useTCP = settings.GetValue(std::string("use_tcp"), std::string("false"));
    dlg.UseTCP->SetValue(Utils::boolFromString(useTCP));

    string disableGL = settings.GetValue(std::string("disable_gl_preview"), std::string("false"));
    dlg.DisableGL->SetValue(Utils::boolFromString(disableGL));

    dlg.HwDevice->Append(wxT("none"), new ClientDataHWDisplay("none", NULL, -1));
    dlg.HwDevice->Select(0u);

    string currentHWDevice = settings.GetValue(std::string("hw_display"), std::string("none"));
    string currentAudioDevice = settings.GetValue(std::string("audio_playback"), std::string("none"));
    int deviceIndex = 0u;

    for(int i = 0; i < display_get_device_count(); i++) {
        display_type_t  *dev = display_get_device_details(i);
        struct display_device *it = dev->devices;

        while(it->name != NULL) {
            deviceIndex += 1;
            ClientDataHWDisplay *data = new ClientDataHWDisplay(it->driver_identifier, it->modes, it->modes_count);
            dlg.HwDevice->Append(wxString::FromUTF8(it->name), data);
            if(strcmp(it->driver_identifier, currentHWDevice.c_str()) == 0) {
                dlg.HwDevice->Select(deviceIndex);
            }

            ++it;
        }
    }

    deviceIndex = 0u;
    for(int i = 0; i < audio_playback_get_device_count(); ++i) {
        audio_playback_type *it = audio_playback_get_device_details(i);;

        while(it->name != NULL) {
            ClientDataCStr *data = new ClientDataCStr(it->driver_identifier);
            dlg.AudioDevice->Append(wxString::FromUTF8(it->name), data);

            if(strcmp(it->driver_identifier, currentAudioDevice.c_str()) == 0) {
                dlg.AudioDevice->Select(deviceIndex);
            }

            deviceIndex += 1;
            ++it;
        }
    }

    if ( dlg.ShowModal() == wxID_OK ) {
        settings.SetValue("use_tcp", dlg.UseTCP->GetValue() ? "true" : "false");
        settings.SetValue("hw_display", dynamic_cast<ClientDataHWDisplay *>(dlg.HwDevice->GetClientObject(dlg.HwDevice->GetSelection()))->identifier);
        {
            string modeStr;
            if(!dlg.HwFormat->HasClientObjectData()) {
                modeStr = "auto";
            } else {
                struct video_desc *mode = (struct video_desc *)
                    dynamic_cast<ClientDataWeakGenericPtr *>(dlg.HwFormat->GetClientObject(dlg.HwDevice->GetSelection()))->get();
                modeStr = Utils::VideoDescSerialize(mode);
            }
            settings.SetValue("hw_display_prefs", modeStr);
        }
        settings.SetValue("disable_gl_preview", dlg.DisableGL->GetValue() ? "true" : "false");
        settings.SetValue("audio_playback", dynamic_cast<ClientDataCStr *>(dlg.AudioDevice->GetClientObject(dlg.AudioDevice->GetSelection()))->get());
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
    About dlg(this);

    dlg.ShowModal();
    /*wxString msg = wxbuildinfo(long_f);
    wxMessageBox(msg, _("Welcome to..."));*/
}

void client_guiFrame::OnListBox1Select1(wxCommandEvent& event)
{
}

void client_guiFrame::OnButton1Click(wxCommandEvent& event)
{


}

void client_guiFrame::OnStopBtnClick(wxCommandEvent& event)
{
    if(player.GetState() == sInit) {
        // nothing
    } else if(player.GetState() == sReady) {
        Stop();
    } else {
        Stop();
    }
}

void client_guiFrame::PlaySelection()
{
    if(this->playList.empty()) {
        StatusBar1->PushStatusText(wxT("No video selected!"));
        //wxMessageBox(_("No video selected"), _("Unable to play"));
        return;
    }

    try {
        double fps;
        if(!this->fps->GetValue().ToDouble(&fps)) {
             /* error! */
        }

        int start_pos = Slider1->GetValue() / 100.0 * (this->playList[0].total_frames - 1);
        player.Play(this->playList[0], fps, start_pos);

        StatusBar1->PushStatusText(this->playList[0].URL);
        ChangeState(sPlaying);
        total_frames = this->playList[0].total_frames;
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Playback error"));
    }
}

void client_guiFrame::Stop()
{
    try {
        player.StopPlayback();
    } catch (std::exception &e) {
        wxMessageBox(wxString::FromUTF8(e.what()), _("Error stopping media"));
    }

    StatusBar1->PushStatusText(wxT("stopped"));

    ChangeState(sInit);
    DoUpdateCounters(0);
    //Refresh();
    gl->LoadSplashScreen();

    total_frames = 0;
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

    if(selectVideo->ShowModal() == wxID_OK ) {
        if(player.GetState() != sInit) {
            this->Stop();
        }
        this->playList.Empty();
        this->playList = selectVideo->GetSelectedVideo();
        if(!this->playList.IsEmpty()) {
            fps->SetValue(wxString::Format(wxT("%2.2f"), this->playList[0].fps));
        }
    }

    /*selectVideo->ListBox1->GetSelections(playListIndices);

    for (int i = 0; i < playListIndices.Count(); ++i) {
        this->playList.Insert(selectVideo->ListBox1->GetString(playListIndices[i]), i);
    }*/
}

void client_guiFrame::PushDisconnect(wxCommandEvent& event)
{
    DoDisconnect();
}

void client_guiFrame::DoDisconnect()
{
    StatusBar1->PushStatusText(wxT("interrupted by server"));
    player.StopPlayback();
    ChangeState(sInit);
}

void client_guiFrame::OnTextCtrl1Text(wxCommandEvent& event)
{
}

void client_guiFrame::DoUpdateCounters(int val)
{
    if(!this->playList.IsEmpty()) {
        Slider1->SetValue((double)  val / (player.GetTotalFrames()- 1) * 100.0);

        FrameCount->SetValue(val);
    } else {
        Slider1->SetValue(0);
        FrameCount->SetValue(0);
    }
}

void client_guiFrame::UpdateTimer(int val)
{
    if(player.GetState() != sInit)
        DoUpdateCounters(val);
}

void client_guiFrame::JumpToFrame(int frame)
{
    try {
        if(frame < 0 || frame >= this->total_frames)
            return;

        if(player.GetState() == sPlaying) {
            player.JumpAndPlay(frame);
        } else if (player.GetState() == sReady) {
            player.JumpAndPause(frame);
        } else {
            return;
        }
    } catch (std::exception &e) {
        wxMessageBox(wxString::FromUTF8(e.what()), _("Error downloading media file"));
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
    if(player.GetState() == sInit && event.GetExtraLong() != MOUSE_CLICKED_MAGIC) {
        PlaySelection();
    } else if(player.GetState() == sPlaying) {
        DoPause();
    } else if(player.GetState() == sReady) {
        Resume();
    }
}

void client_guiFrame::DoPause()
{
    if(player.GetState() != sPlaying)
        return;

    try {
        player.Pause();
        ChangeState(sReady);
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error pausing media"));
    }
}

void client_guiFrame::Resume()
{
    try {
        // TODO: handle TMOUT
        player.Play();

        ChangeState(sPlaying);
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error downloading media file"));
    }
}

void client_guiFrame::ChangeState(enum playerState newState)
{
    player.ChangeState(newState);

    switch(newState) {
        case sInit:
            ResetToDefaultValues();
            Pause->SetLabel(wxT("▶"));
            break;
        case sReady:
            Pause->SetLabel(wxT("▶"));
            break;
        case sPlaying:
            Pause->SetLabel(wxT("❚❚"));
            break;
    }
}

void client_guiFrame::OnButton1Click2(wxCommandEvent& event)
{
    wxString msgstr;
    struct message msg;
    struct response resp;

    try {
        double fps;
        if(!this->fps->GetValue().ToDouble(&fps)) {
             /* error! */
        }

        player.SetFPS(fps);
    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error to set fps"));
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
#ifndef __WXMAC__
        FlexGridSizer2->Show(true);
#endif
        StatusBar1->Show(true);
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
#ifndef __WXMAC__
        FlexGridSizer2->Show(false);
#endif
        StatusBar1->Show(false);
        ShowFullScreen(true, wxFULLSCREEN_ALL);
    }
}

void client_guiFrame::TogglePause(wxCommandEvent& evt)
{
    OnPauseClick(evt);
}

void client_guiFrame::Mouse(wxMouseEvent& evt)
{

    if(evt.GetEventObject() == gl) {
        if(evt.LeftUp()) {
            dragging = false;
        } else if(evt.Dragging()) {
            if(player.GetState() != sInit) {
                wxPoint position = evt.GetPosition();
                if(dragging) {
                    gl->GoPixels(lastDragPosition.x - position.x, lastDragPosition.y - position.y);
                }
                lastDragPosition = position;
                dragging = true;
            }
        } else { /* motion */
#ifndef __WXMAC__
            if(IsFullScreen()  && !FlexGridSizer1->IsShown(FlexGridSizer2) && evt.GetY() > (gl->GetSize().y - 5)) {
                //std::cerr << "." << evt.GetY();
                FlexGridSizer2->Show(true);
                FlexGridSizer1->Layout();
                this->Fit();
            } else if(IsFullScreen() && FlexGridSizer1->IsShown(FlexGridSizer2)  && evt.GetY() < (gl->GetSize().y - 20)) {
                //std::cerr << "!" << evt.GetY();
                FlexGridSizer2->Show(false);
                FlexGridSizer1->Layout();
                this->Fit();
            }
#endif
        }
    }
}

void client_guiFrame::OnFrameCountChange(wxSpinEvent& event)
{
    if(player.GetState() != sInit) {
        JumpToFrame(event.GetPosition());
    }
}

void client_guiFrame::KeyDown(wxKeyEvent& evt)
{
    switch(evt.GetKeyCode()) {
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
    }

    if(player.GetState() != sInit) {
        switch(evt.GetKeyCode()) {
            case WXK_SPACE:
                 if(player.GetState() == sPlaying) {
                    DoPause();
                 } else if(player.GetState() == sReady) {
                    Resume();
                 }
                break;
            case 'K':
                if(player.GetState() == sPlaying)
                    DoPause();
                break;
            case 'J':
                JumpToFrame(player.GetCurrentFrame() - 1);
                break;
            case 'L':
                JumpToFrame(player.GetCurrentFrame() + 1);
                break;
            case 'C':
                gl->ToggleLightness();
                if(player.GetState() == sReady)
                    gl->Render(true);
                break;
            case 'R':
            case 'G':
            case 'B':
                if(LastColorModifiingKey == evt.GetUnicodeKey() && LastColorModifiingModifiers == evt.GetModifiers()) {
                    gl->DefaultLightness();
                    LastColorModifiingModifiers = LastColorModifiingKey = 0;
                } else {
                    if(evt.GetModifiers() == wxMOD_CONTROL) {
                        gl->HideChannel(evt.GetUnicodeKey());
                    } else {
                        gl->ShowOnlyChannel(evt.GetUnicodeKey());
                    }
                    LastColorModifiingKey = evt.GetUnicodeKey();
                    LastColorModifiingModifiers = evt.GetModifiers();
                }
                if(player.GetState() == sReady)
                    gl->Render(true);

                break;
            case 'P':
                ToggleLoop->SetValue(!ToggleLoop->GetValue());
                break;
            case '+':
                gl->Zoom(1/0.8);
                break;
            case '-':
                gl->Zoom(0.8);
                break;
            case WXK_LEFT:
                gl->Go(-0.03, 0);
                break;
            case WXK_UP:
                gl->Go(0, -0.03);
                break;
            case WXK_RIGHT:
                gl->Go(0.03, 0);
                break;
            case WXK_DOWN:
                gl->Go(0, 0.03);
                break;
            default:
                std::cerr << "Unknown key: " << evt.GetUnicodeKey() << std::endl;
        }
    }
}


void client_guiFrame::OnButton1Click3(wxCommandEvent& event)
{
    try {
        player.SetLoop(ToggleLoop->GetValue());
    } catch (std::exception &e) {
        ToggleLoop->SetValue(!ToggleLoop->GetValue());
        wxMessageBox(wxString::FromUTF8(e.what()), _("Error setting loop"));
    }
}

void client_guiFrame::Wheel(wxMouseEvent& evt)
{
    if(evt.GetEventObject() == gl) {
        if(player.GetState() != sInit) {
            if(evt.GetWheelRotation() == 0)
                return;
            double ratio = evt.GetWheelRotation() / 200.0;
            ratio = ratio > 0? ratio : ratio / 2.0;
            gl->Zoom(ratio);
        }
    }
}

void client_guiFrame::ChangeSpeed(double ratio)
{
    try {
        player.SetSpeed(player.GetSpeed() * ratio);
    } catch (std::exception &e) {
        player.SetSpeed(player.GetSpeed() / ratio);
        wxMessageBox(wxString::FromUTF8(e.what()), _("Error setting speed"));
    }
    SpeedStr->SetLabel(Utils::FromCDouble(player.GetSpeed(), 2));
}

void client_guiFrame::ChangeDirection(int direction)
{
    try {
        player.SetSpeed(1.0 * direction);
    } catch (std::exception &e) {
        wxMessageBox(wxString::FromUTF8(e.what()), _("Error setting speed"));
    }

    SpeedStr->SetLabel(Utils::FromCDouble(player.GetSpeed(), 2));
}


void client_guiFrame::OnForwardFastClick(wxCommandEvent& event)
{
    ChangeSpeed(2.0);
}

void client_guiFrame::OnForwardSlowClick(wxCommandEvent& event)
{
    ChangeDirection(1);
}

void client_guiFrame::OnBackwardSlowClick(wxCommandEvent& event)
{
    ChangeDirection(-1);
}

void client_guiFrame::OnBackwardFastClick(wxCommandEvent& event)
{
    ChangeSpeed(0.5);
}

void client_guiFrame::ResetToDefaultValues()
{
    gl->ResetDefaults();
    ChangeSpeed(1.0);
    ChangeDirection(1);
}

void client_guiFrame::OnKeyBindingsHelp(wxCommandEvent& event)
{
    KeyBindingsHelp dlg(this);

    dlg.ShowModal();
}

void client_guiFrame::OnHDToggle(wxCommandEvent& event)
{
    player.SetHDDownscaling(HD->GetValue());
}
