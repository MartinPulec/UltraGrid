#include "VideoProperties.h"

//(*InternalHeaders(VideoProperties)
#include <wx/button.h>
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(VideoProperties)
const long VideoProperties::ID_STATICTEXT2 = wxNewId();
const long VideoProperties::ID_STATICTEXT1 = wxNewId();
const long VideoProperties::ID_TEXTCTRL1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(VideoProperties,wxDialog)
	//(*EventTable(VideoProperties)
	//*)
END_EVENT_TABLE()

VideoProperties::VideoProperties(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(VideoProperties)
	wxFlexGridSizer* FlexGridSizer1;
	wxFlexGridSizer* FlexGridSizer2;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;

	Create(parent, wxID_ANY, _("Video Properties"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("wxID_ANY"));
	SetClientSize(wxSize(441,107));
	FlexGridSizer1 = new wxFlexGridSizer(0, 1, 0, 0);
	Name = new wxStaticText(this, ID_STATICTEXT2, _("Label"), wxDefaultPosition, wxSize(388,15), 0, _T("ID_STATICTEXT2"));
	FlexGridSizer1->Add(Name, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer2 = new wxFlexGridSizer(0, 2, 0, 0);
	FlexGridSizer2->AddGrowableCol(1);
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Preferred FPS"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	FlexGridSizer2->Add(StaticText1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FPSCtrl = new wxTextCtrl(this, ID_TEXTCTRL1, _("30.00"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_TEXTCTRL1"));
	FlexGridSizer2->Add(FPSCtrl, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer1->Add(FlexGridSizer2, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StdDialogButtonSizer1 = new wxStdDialogButtonSizer();
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_OK, wxEmptyString));
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_CANCEL, wxEmptyString));
	StdDialogButtonSizer1->Realize();
	FlexGridSizer1->Add(StdDialogButtonSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	SetSizer(FlexGridSizer1);
	FlexGridSizer1->SetSizeHints(this);
	//*)
}

VideoProperties::~VideoProperties()
{
	//(*Destroy(VideoProperties)
	//*)
}

