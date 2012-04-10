#include "CompressionSetting.h"

//(*InternalHeaders(CompressionSetting)
#include <wx/button.h>
#include <wx/string.h>
#include <wx/intl.h>
//*)

#include <iostream>

//(*IdInit(CompressionSetting)
const long CompressionSetting::ID_STATICTEXT1 = wxNewId();
const long CompressionSetting::ID_CHOICE1 = wxNewId();
const long CompressionSetting::ID_SPINCTRL1 = wxNewId();
const long CompressionSetting::ID_STATICTEXT2 = wxNewId();
const long CompressionSetting::ID_STATICTEXT3 = wxNewId();
//*)

BEGIN_EVENT_TABLE(CompressionSetting,wxDialog)
	//(*EventTable(CompressionSetting)
	//*)
END_EVENT_TABLE()

CompressionSetting::CompressionSetting(wxWindow* parent)
{
	//(*Initialize(CompressionSetting)
	wxFlexGridSizer* FlexGridSizer1;
	wxFlexGridSizer* FlexGridSizer2;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;
	
	Create(parent, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("wxID_ANY"));
	FlexGridSizer1 = new wxFlexGridSizer(0, 1, 0, 0);
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Select preferred compression:"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	FlexGridSizer1->Add(StaticText1, 1, wxALL|wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer2 = new wxFlexGridSizer(0, 3, 0, 0);
	Choice1 = new wxChoice(this, ID_CHOICE1, wxDefaultPosition, wxSize(163,29), 0, 0, 0, wxDefaultValidator, _T("ID_CHOICE1"));
	FlexGridSizer2->Add(Choice1, 1, wxALL|wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL, 5);
	SpinCtrl1 = new wxSpinCtrl(this, ID_SPINCTRL1, _T("80"), wxDefaultPosition, wxSize(56,25), 0, 1, 100, 80, _T("ID_SPINCTRL1"));
	SpinCtrl1->SetValue(_T("80"));
	SpinCtrl1->Hide();
	FlexGridSizer2->Add(SpinCtrl1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer1->Add(FlexGridSizer2, 1, wxALL|wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL, 5);
	StaticText2 = new wxStaticText(this, ID_STATICTEXT2, _("Description"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_STATICTEXT2"));
	FlexGridSizer1->Add(StaticText2, 1, wxALL|wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL, 5);
	description = new wxStaticText(this, ID_STATICTEXT3, _("description"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_STATICTEXT3"));
	FlexGridSizer1->Add(description, 1, wxALL|wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL, 5);
	StdDialogButtonSizer1 = new wxStdDialogButtonSizer();
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_OK, wxEmptyString));
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_CANCEL, wxEmptyString));
	StdDialogButtonSizer1->Realize();
	FlexGridSizer1->Add(StdDialogButtonSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	SetSizer(FlexGridSizer1);
	FlexGridSizer1->Fit(this);
	FlexGridSizer1->SetSizeHints(this);
	
	Connect(ID_CHOICE1,wxEVT_COMMAND_CHOICE_SELECTED,(wxObjectEventFunction)&CompressionSetting::OnChoice1Select);
	//*)

	Choice1->Append(wxT("none"));
	Choice1->Append(wxT("DXT1"));
	Choice1->Append(wxT("DXT5"));
	Choice1->Append(wxT("JPEG"));
	Choice1->Select(0);
	compDesc[wxT("none")] = wxT("Send uncompressed video stream.");
	compDesc[wxT("DXT1")] = wxT("Provides simple, yet efficient compression.");
	compDesc[wxT("DXT5")] = wxT("Improved version of DXT1 with better PSNR but higher bandwidth");
	compDesc[wxT("JPEG")] = wxT("Provides variable bandwidth JPEG compression");

	description->SetLabel(compDesc[Choice1->GetString(Choice1->GetSelection())]);
    FlexGridSizer1->SetSizeHints(this);
	FlexGridSizer1->Fit(this);
	sizer = FlexGridSizer1;
}

CompressionSetting::~CompressionSetting()
{
	//(*Destroy(CompressionSetting)
	//*)
}


void CompressionSetting::OnRadioButton1Select(wxCommandEvent& event)
{
}

void CompressionSetting::OnInit(wxInitDialogEvent& event)
{
}

void CompressionSetting::OnChoice1Select(wxCommandEvent& event)
{
    SelectValue(Choice1->GetSelection());
}

void CompressionSetting::SetValue(wxString val)
{
    for (int i = 0; i < Choice1->GetCount(); ++i) {
        if(val == Choice1->GetString(i)) {
            SelectValue(i);
        }
    }
}

void CompressionSetting::SelectValue(int i)
{
    description->SetLabel(compDesc[Choice1->GetString(i)]);
    description->Wrap(163);

    if(Choice1->GetString(i) == wxT("JPEG")) {
        SpinCtrl1->Show();
    } else {
        SpinCtrl1->Hide();
    }

    Choice1->Select(i);

    sizer->SetSizeHints(this);
	sizer->Fit(this);
}
