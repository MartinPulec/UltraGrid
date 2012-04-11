#ifndef ABOUT_H
#define ABOUT_H

//(*Headers(About)
#include <wx/dialog.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/statbmp.h>
//*)

class About: public wxDialog
{
	public:

		About(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~About();

		//(*Declarations(About)
		wxStaticText* StaticText1;
		wxStaticBitmap* BitmapFINT;
		wxStaticBitmap* StaticBitmap1;
		//*)

	protected:

		//(*Identifiers(About)
		static const long ID_STATICBITMAP1;
		static const long ID_BITMAPFINT;
		static const long ID_STATICTEXT1;
		//*)

	private:

		//(*Handlers(About)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
