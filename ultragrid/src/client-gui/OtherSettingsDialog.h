#ifndef OTHERSETTINGSDIALOG_H
#define OTHERSETTINGSDIALOG_H

//(*Headers(OtherSettingsDialog)
#include <wx/checkbox.h>
#include <wx/dialog.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/choice.h>
//*)

class OtherSettingsDialog: public wxDialog
{
	public:

		OtherSettingsDialog(wxWindow* parent,wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~OtherSettingsDialog();

		//(*Declarations(OtherSettingsDialog)
		wxChoice* HwDevice;
		wxCheckBox* UseTCP;
		wxStaticText* HwDeviceLabel;
		//*)

	protected:

		//(*Identifiers(OtherSettingsDialog)
		static const long ID_HWDEVICELABEL;
		static const long ID_HWDEV;
		static const long ID_USETCP;
		//*)

	private:

		//(*Handlers(OtherSettingsDialog)
		void OnHwDeviceSelect(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
