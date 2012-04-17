#include "OtherSettingsDialog.h"

//(*InternalHeaders(OtherSettingsDialog)
#include <wx/button.h>
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(OtherSettingsDialog)
const long OtherSettingsDialog::ID_USETCP = wxNewId();
//*)

BEGIN_EVENT_TABLE(OtherSettingsDialog,wxDialog)
	//(*EventTable(OtherSettingsDialog)
	//*)
END_EVENT_TABLE()

OtherSettingsDialog::OtherSettingsDialog(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(OtherSettingsDialog)
	wxFlexGridSizer* FlexGridSizer1;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;
	
	Create(parent, wxID_ANY, _("Other Settings"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("wxID_ANY"));
	FlexGridSizer1 = new wxFlexGridSizer(2, 1, 0, 0);
	UseTCP = new wxCheckBox(this, ID_USETCP, _("Use TCP"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_USETCP"));
	UseTCP->SetValue(false);
	UseTCP->SetHelpText(_("Use TCP for data transmit (slow in most circumstances)"));
	FlexGridSizer1->Add(UseTCP, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StdDialogButtonSizer1 = new wxStdDialogButtonSizer();
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_OK, wxEmptyString));
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_CANCEL, wxEmptyString));
	StdDialogButtonSizer1->Realize();
	FlexGridSizer1->Add(StdDialogButtonSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	SetSizer(FlexGridSizer1);
	FlexGridSizer1->Fit(this);
	FlexGridSizer1->SetSizeHints(this);
	//*)
}

OtherSettingsDialog::~OtherSettingsDialog()
{
	//(*Destroy(OtherSettingsDialog)
	//*)
}

