#ifndef OTHERSETTINGSDIALOG_H
#define OTHERSETTINGSDIALOG_H

//(*Headers(OtherSettingsDialog)
#include <wx/checkbox.h>
#include <wx/dialog.h>
#include <wx/sizer.h>
//*)

class OtherSettingsDialog: public wxDialog
{
	public:

		OtherSettingsDialog(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~OtherSettingsDialog();

		//(*Declarations(OtherSettingsDialog)
		wxCheckBox* UseTCP;
		//*)

	protected:

		//(*Identifiers(OtherSettingsDialog)
		static const long ID_USETCP;
		//*)

	private:

		//(*Handlers(OtherSettingsDialog)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
