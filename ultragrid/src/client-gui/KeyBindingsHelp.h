#ifndef KEYBINDINGSHELP_H
#define KEYBINDINGSHELP_H

//(*Headers(KeyBindingsHelp)
#include <wx/dialog.h>
#include <wx/sizer.h>
#include <wx/html/htmlwin.h>
//*)

class KeyBindingsHelp: public wxDialog
{
	public:

		KeyBindingsHelp(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~KeyBindingsHelp();

		//(*Declarations(KeyBindingsHelp)
		wxHtmlWindow* HtmlWindow1;
		//*)

	protected:

		//(*Identifiers(KeyBindingsHelp)
		static const long ID_HTMLWINDOW1;
		//*)

	private:

		//(*Handlers(KeyBindingsHelp)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
