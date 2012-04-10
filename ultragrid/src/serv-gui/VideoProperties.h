#ifndef VIDEOPROPERTIES_H
#define VIDEOPROPERTIES_H

//(*Headers(VideoProperties)
#include <wx/dialog.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
//*)

class VideoProperties: public wxDialog
{
	public:

		VideoProperties(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~VideoProperties();

		//(*Declarations(VideoProperties)
		wxStaticText* Name;
		wxStaticText* StaticText1;
		wxTextCtrl* FPSCtrl;
		//*)

	protected:

		//(*Identifiers(VideoProperties)
		static const long ID_STATICTEXT2;
		static const long ID_STATICTEXT1;
		static const long ID_TEXTCTRL1;
		//*)

	private:

		//(*Handlers(VideoProperties)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
