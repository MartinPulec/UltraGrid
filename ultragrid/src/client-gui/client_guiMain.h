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

#include "include/Settings.h"
#include "include/UGReceiver.h"
#include "include/GLView.h"
#include "include/AsyncMsgHandler.h"
#include "include/VideoEntry.h"
#include "VideoSelection.h"
#include "include/ProgressSlider.h"
#include "include/CustomGridBagSizer.h"
#include "include/ClientManager.h"

//(*Headers(client_guiFrame)
#include <wx/glcanvas.h>
#include <wx/spinctrl.h>
#include <wx/sizer.h>
#include <wx/button.h>
#include <wx/menu.h>
#include <wx/slider.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
//*)

#include <wx/tglbtn.h>

enum playerState {
    sInit,
    sReady,
    sPlaying
};

class client_guiFrame: public wxFrame
{
    public:

        client_guiFrame(wxWindow* parent,wxWindowID id = -1);
        virtual ~client_guiFrame();
        void DataReceived();
        void DoDisconnect();

    private:

        //(*Handlers(client_guiFrame)
        void OnQuit(wxCommandEvent& event);
        void OnAbout(wxCommandEvent& event);
        void OnServerSetting(wxCommandEvent& event);
        void OnCompressSetting(wxCommandEvent& event);
        void On3Box1Select(wxCommandEvent& event);
        void OnListBox1Select1(wxCommandEvent& event);
        void OnButton1Click(wxCommandEvent& event);
        void OnButton1Click1(wxCommandEvent& event);
        void OnSelectClick(wxCommandEvent& event);
        void OnTextCtrl1Text(wxCommandEvent& event);
        void OnPauseClick(wxCommandEvent& event);
        void OnButton1Click2(wxCommandEvent& event);
        void OnGLCanvas1Paint(wxPaintEvent& event);
        void OnFrameCountChange(wxSpinEvent& event);
        void OnStopBtnClick(wxCommandEvent& event);
        void OnButton1Click3(wxCommandEvent& event);
        //*)
        void Resize(wxCommandEvent&);
        void UpdateTimer(wxCommandEvent&);
        void Scrolled(wxCommandEvent&);
        void ToggleFullscreen(wxCommandEvent&);
        void TogglePause(wxCommandEvent&);
        void Mouse(wxMouseEvent& evt);
        void KeyDown(wxKeyEvent& evt);
        void Wheel(wxMouseEvent& evt);

        void PlaySelection();
        void Stop();
        void DoPause();
        void Resume();
        void JumpToFrame(int frame);
        void DoUpdateCounters(int val);

        int FilterEvent(wxEvent& event);

        void ChangeState(enum playerState);

        //(*Identifiers(client_guiFrame)
        static const long ID_GLCANVAS1;
        static const long ID_BUTTON2;
        static const long ID_TEXTCTRL1;
        static const long ID_BUTTON3;
        static const long ID_FR_LABEL;
        static const long ID_FR;
        static const long ID_ToggleLoop;
        static const long ID_SLIDER1;
        static const long PlayButton;
        static const long ID_BUTTON1;
        static const long idMenuQuit;
        static const long idServerSetting;
        static const long idCompressionSetting;
        static const long idMenuAbout;
        static const long ID_STATUSBAR1;
        //*)

        //(*Declarations(client_guiFrame)
        wxFlexGridSizer* FlexGridSizer2;
        wxToggleButton* ToggleLoop;
        wxStaticText* FrameCountLabel;
        wxStatusBar* StatusBar1;
        wxButton* Select;
        wxButton* FPSOk;
        wxTextCtrl* fps;
        GLView* gl;
        ProgressSlider* Slider1;
        wxButton* StopBtn;
        wxMenuItem* MenuItem4;
        CustomGridBagSizer* FlexGridSizer1;
        wxButton* Pause;
        wxSpinCtrl* FrameCount;
        //*)

        ClientManager connection;
        ArrayOfVideoEntries playList;
        Settings settings;

        UGReceiver *receiver;
        VideoSelection *selectVideo;
        AsyncMsgHandler msgHandler;

        long int total_frames;
        int LastColorModifiingKey;
        int LastColorModifiingModifiers;

        bool dragging;
        wxPoint lastDragPosition;

        enum playerState state;

        DECLARE_EVENT_TABLE()
    friend class UltraGridManager;
};

#endif // CLIENT_GUIMAIN_H
