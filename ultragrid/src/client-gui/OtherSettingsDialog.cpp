#include "OtherSettingsDialog.h"

//(*InternalHeaders(OtherSettingsDialog)
#include <wx/button.h>
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(OtherSettingsDialog)
const long OtherSettingsDialog::ID_HWDEVICELABEL = wxNewId();
const long OtherSettingsDialog::ID_HWDEV = wxNewId();
const long OtherSettingsDialog::ID_USETCP = wxNewId();
const long OtherSettingsDialog::ID_STATICTEXT1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(OtherSettingsDialog,wxDialog)
	//(*EventTable(OtherSettingsDialog)
	//*)
END_EVENT_TABLE()

OtherSettingsDialog::OtherSettingsDialog(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(OtherSettingsDialog)
	wxFlexGridSizer* FlexGridSizer1;
	wxFlexGridSizer* FlexGridSizer2;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;

	Create(parent, wxID_ANY, _("Other Settings"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("wxID_ANY"));
	SetClientSize(wxSize(317,203));
	FlexGridSizer1 = new wxFlexGridSizer(4, 1, 0, 0);
	FlexGridSizer2 = new wxFlexGridSizer(1, 2, 0, 0);
	HwDeviceLabel = new wxStaticText(this, ID_HWDEVICELABEL, _("Output device:"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_HWDEVICELABEL"));
	FlexGridSizer2->Add(HwDeviceLabel, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	HwDevice = new wxChoice(this, ID_HWDEV, wxDefaultPosition, wxSize(214,29), 0, 0, 0, wxDefaultValidator, _T("ID_HWDEV"));
	FlexGridSizer2->Add(HwDevice, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer1->Add(FlexGridSizer2, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	UseTCP = new wxCheckBox(this, ID_USETCP, _("Use TCP"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_USETCP"));
	UseTCP->SetValue(false);
	UseTCP->SetHelpText(_("Use TCP for data transmit (slow in most circumstances)"));
	FlexGridSizer1->Add(UseTCP, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StdDialogButtonSizer1 = new wxStdDialogButtonSizer();
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_OK, wxEmptyString));
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_CANCEL, wxEmptyString));
	StdDialogButtonSizer1->Realize();
	FlexGridSizer1->Add(StdDialogButtonSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Note that changing these options requires program restart."), wxDefaultPosition, wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	FlexGridSizer1->Add(StaticText1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	SetSizer(FlexGridSizer1);
	FlexGridSizer1->SetSizeHints(this);
	//*)
}

OtherSettingsDialog::~OtherSettingsDialog()
{
	//(*Destroy(OtherSettingsDialog)
	//*)
}

