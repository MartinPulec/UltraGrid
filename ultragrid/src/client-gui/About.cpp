#include "About.h"

//(*InternalHeaders(About)
#include <wx/button.h>
#include <wx/string.h>
#include <wx/intl.h>
//*)

#include <wx/image.h>
#include <wx/mstream.h>

#include "cesnet-logo-png.h"

//(*IdInit(About)
const long About::ID_STATICBITMAP1 = wxNewId();
const long About::ID_STATICTEXT1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(About,wxDialog)
	//(*EventTable(About)
	//*)
END_EVENT_TABLE()

About::About(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(About)
	wxFlexGridSizer* FlexGridSizer1;
	wxFlexGridSizer* FlexGridSizer2;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;

	Create(parent, id, _("Welcome to..."), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxDefaultSize);
	Move(wxDefaultPosition);
	FlexGridSizer1 = new wxFlexGridSizer(0, 1, 0, 0);
	FlexGridSizer2 = new wxFlexGridSizer(0, 2, 0, 0);
	StaticBitmap1 = new wxStaticBitmap(this, ID_STATICBITMAP1, wxNullBitmap, wxDefaultPosition, wxSize(128,59), 0, _T("ID_STATICBITMAP1"));
	FlexGridSizer2->Add(StaticBitmap1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("FlashNET is a remote preview tool\njointly developed by CESNET and\nFINT."), wxDefaultPosition, wxDefaultSize, 0, _T("ID_STATICTEXT1"));
	FlexGridSizer2->Add(StaticText1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer1->Add(FlexGridSizer2, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StdDialogButtonSizer1 = new wxStdDialogButtonSizer();
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_OK, wxEmptyString));
	StdDialogButtonSizer1->Realize();
	FlexGridSizer1->Add(StdDialogButtonSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	SetSizer(FlexGridSizer1);
	FlexGridSizer1->Fit(this);
	FlexGridSizer1->SetSizeHints(this);
	//*)

    wxMemoryInputStream in((const unsigned char*) cesnet_logo_png, sizeof(cesnet_logo_png));

    wxImage cesnet_(in);
    StaticBitmap1->SetBitmap(wxBitmap(cesnet_));
}

About::~About()
{
	//(*Destroy(About)
	//*)
}

