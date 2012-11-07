#include "OtherSettingsDialog.h"

//(*InternalHeaders(OtherSettingsDialog)
#include <wx/button.h>
#include <wx/string.h>
#include <wx/intl.h>
//*)

#include "ClientDataHWDisplay.h"
#include "ClientDataWeakGenericPtr.h"
#include "video.h"

//(*IdInit(OtherSettingsDialog)
const long OtherSettingsDialog::ID_HWDEVICELABEL = wxNewId();
const long OtherSettingsDialog::ID_HWDEV = wxNewId();
const long OtherSettingsDialog::ID_STATICTEXT2 = wxNewId();
const long OtherSettingsDialog::ID_HWMODE = wxNewId();
const long OtherSettingsDialog::ID_USETCP = wxNewId();
const long OtherSettingsDialog::ID_CHECKBOX1 = wxNewId();
const long OtherSettingsDialog::ID_PROGRESET = wxNewId();
//*)

BEGIN_EVENT_TABLE(OtherSettingsDialog,wxDialog)
    EVT_CHOICE(ID_HWDEV, OtherSettingsDialog::OnHwDeviceSelect)
	//(*EventTable(OtherSettingsDialog)
	//*)
END_EVENT_TABLE()

OtherSettingsDialog::OtherSettingsDialog(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(OtherSettingsDialog)
	wxFlexGridSizer* FlexGridSizer1;
	wxFlexGridSizer* FlexGridSizer2;
	wxFlexGridSizer* FlexGridSizer3;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;

	Create(parent, wxID_ANY, _("Other Settings"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("wxID_ANY"));
	SetClientSize(wxSize(317,203));
	FlexGridSizer1 = new wxFlexGridSizer(4, 1, 0, 0);
	FlexGridSizer2 = new wxFlexGridSizer(2, 2, 0, 0);
	HwDeviceLabel = new wxStaticText(this, ID_HWDEVICELABEL, _("Output device*:"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_HWDEVICELABEL"));
	FlexGridSizer2->Add(HwDeviceLabel, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	HwDevice = new wxChoice(this, ID_HWDEV, wxDefaultPosition, wxSize(214,29), 0, 0, 0, wxDefaultValidator, _T("ID_HWDEV"));
	FlexGridSizer2->Add(HwDevice, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	VideoFormatForce = new wxStaticText(this, ID_STATICTEXT2, _("Force video format*:"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	FlexGridSizer2->Add(VideoFormatForce, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	HwFormat = new wxChoice(this, ID_HWMODE, wxDefaultPosition, wxSize(214,29), 0, 0, 0, wxDefaultValidator, _T("ID_HWMODE"));
	HwFormat->SetSelection( HwFormat->Append(_("auto")) );
	FlexGridSizer2->Add(HwFormat, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer1->Add(FlexGridSizer2, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer3 = new wxFlexGridSizer(1, 2, 0, 0);
	UseTCP = new wxCheckBox(this, ID_USETCP, _("Use TCP*"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_USETCP"));
	UseTCP->SetValue(false);
	UseTCP->SetHelpText(_("Use TCP for data transmit (slow in most circumstances)"));
	FlexGridSizer3->Add(UseTCP, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	DisableGL = new wxCheckBox(this, ID_CHECKBOX1, _("Disable GL Preview*"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_CHECKBOX1"));
	DisableGL->SetValue(false);
	FlexGridSizer3->Add(DisableGL, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer1->Add(FlexGridSizer3, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StdDialogButtonSizer1 = new wxStdDialogButtonSizer();
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_OK, wxEmptyString));
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_CANCEL, wxEmptyString));
	StdDialogButtonSizer1->Realize();
	FlexGridSizer1->Add(StdDialogButtonSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	ProgReset = new wxStaticText(this, ID_PROGRESET, _("*Note that changing this option requires program restart."), wxDefaultPosition, wxDefaultSize, 0, _T("ID_PROGRESET"));
	FlexGridSizer1->Add(ProgReset, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	SetSizer(FlexGridSizer1);
	FlexGridSizer1->SetSizeHints(this);
	//*)
}

OtherSettingsDialog::~OtherSettingsDialog()
{
	//(*Destroy(OtherSettingsDialog)
	//*)
}

void OtherSettingsDialog::OnHwDeviceSelect(wxCommandEvent& event)
{
    HwFormat->Clear();
    struct video_desc * modes = dynamic_cast<ClientDataHWDisplay *>(HwDevice->GetClientObject(HwDevice->GetSelection()))->modes;
    ssize_t count = dynamic_cast<ClientDataHWDisplay *>(HwDevice->GetClientObject(HwDevice->GetSelection()))->modes_count;

    HwFormat->Append(_T("auto"));

    for(int i = 0; i < count; ++i) {
        wxString item;
        item << modes[i].width << _T("x") << modes[i].height << _T("@") << modes[i].fps;
        HwFormat->Append(item, new ClientDataWeakGenericPtr((void *) &modes[i]));
    }

    HwFormat->Select(0);
}
