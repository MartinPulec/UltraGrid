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
#include "include/VideoBuffer.h"
#include "include/Player.h"

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

class client_guiFrame: public wxFrame
{
    public:

        client_guiFrame(wxWindow* parent,wxWindowID id = -1);
        virtual ~client_guiFrame();
        void DataReceived();
        void DoDisconnect();
        void UpdateTimer(int val);

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
        void OnForwardFastClick(wxCommandEvent& event);
        void OnForwardSlowClick(wxCommandEvent& event);
        void OnBackwardSlowClick(wxCommandEvent& event);
        void OnBackwardFastClick(wxCommandEvent& event);
        //*)
        void Resize(wxCommandEvent&);
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
        void ChangeDirection(int direction);
        void ChangeSpeed(double ratio);

        int FilterEvent(wxEvent& event);

        void ChangeState(enum playerState);
        void ResetToDefaultValues();

        wxString FromCDouble(double value, int precision);

        //(*Identifiers(client_guiFrame)
        static const long ID_GLCANVAS1;
        static const long ID_BUTTON2;
        static const long ID_TEXTCTRL1;
        static const long ID_BUTTON3;
        static const long ID_FR_LABEL;
        static const long ID_FR;
        static const long ID_ToggleLoop;
        static const long ID_SLIDER1;
        static const long ID_Backward;
        static const long ID_SPEED_STR;
        static const long ID_Forward;
        static const long ID_Slower;
        static const long ID_Quicker;
        static const long PlayButton;
        static const long ID_BUTTON1;
        static const long idMenuQuit;
        static const long idServerSetting;
        static const long idCompressionSetting;
        static const long idMenuAbout;
        static const long ID_STATUSBAR1;
        //*)

        //(*Declarations(client_guiFrame)
        wxButton* Backward;
        wxFlexGridSizer* FlexGridSizer2;
        wxButton* Slower;
        wxToggleButton* ToggleLoop;
        wxStaticText* FrameCountLabel;
        wxStatusBar* StatusBar1;
        wxButton* Select;
        wxButton* FPSOk;
        wxTextCtrl* fps;
        wxStaticText* SpeedStr;
        wxButton* Forward;
        GLView* gl;
        ProgressSlider* Slider1;
        wxButton* StopBtn;
        wxMenuItem* MenuItem4;
        wxButton* Quicker;
        CustomGridBagSizer* FlexGridSizer1;
        wxButton* Pause;
        wxSpinCtrl* FrameCount;
        //*)

        Player player;
        ArrayOfVideoEntries playList;
        Settings settings;

        VideoSelection *selectVideo;
        AsyncMsgHandler msgHandler;

        long int total_frames;
        int LastColorModifiingKey;
        int LastColorModifiingModifiers;

        bool dragging;
        wxPoint lastDragPosition;

        DECLARE_EVENT_TABLE()
    friend class UltraGridManager;
    friend class Player;
};

#endif // CLIENT_GUIMAIN_H
