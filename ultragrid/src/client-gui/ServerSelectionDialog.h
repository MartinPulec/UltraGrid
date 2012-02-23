#ifndef SERVERSELECTIONDIALOG_H
#define SERVERSELECTIONDIALOG_H

//(*Headers(ServerSelectionDialog)
#include <wx/dialog.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
//*)

class ServerSelectionDialog: public wxDialog
{
	public:

		ServerSelectionDialog(wxWindow* parent,wxWindowID id=wxID_ANY);
		virtual ~ServerSelectionDialog();

		//(*Declarations(ServerSelectionDialog)
		wxStaticText* StaticText1;
		wxTextCtrl* TextCtrl1;
		//*)

	protected:

		//(*Identifiers(ServerSelectionDialog)
		static const long ID_STATICTEXT1;
		static const long ID_TEXTCTRL1;
		//*)

	private:

		//(*Handlers(ServerSelectionDialog)
		void OnButton1Click(wxCommandEvent& event);
		void OnButton1Click1(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
