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
		wxCheckBox* DisableGL;
		wxChoice* HwDevice;
		wxChoice* HwFormat;
		wxStaticText* VideoFormatForce;
		wxStaticText* AudioDeviceText;
		wxStaticText* ProgReset;
		wxCheckBox* UseTCP;
		wxChoice* AudioDevice;
		wxStaticText* HwDeviceLabel;
		//*)

	protected:

		//(*Identifiers(OtherSettingsDialog)
		static const long ID_HWDEVICELABEL;
		static const long ID_HWDEV;
		static const long ID_STATICTEXT2;
		static const long ID_HWMODE;
		static const long ID_STATICTEXT1;
		static const long ID_AUDIODEVICE;
		static const long ID_USETCP;
		static const long ID_CHECKBOX1;
		static const long ID_PROGRESET;
		//*)

	private:

		//(*Handlers(OtherSettingsDialog)
		void OnHwDeviceSelect(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
