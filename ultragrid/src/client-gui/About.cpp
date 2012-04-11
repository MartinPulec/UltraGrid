#include "About.h"

//(*InternalHeaders(About)
#include <wx/button.h>
#include <wx/string.h>
#include <wx/intl.h>
//*)

#include <iostream>

#include <wx/image.h>
#include <wx/mstream.h>

#include "cesnet-logo-png.h"
#include "fint-logo-png.h"

//(*IdInit(About)
const long About::ID_STATICBITMAP1 = wxNewId();
const long About::ID_BITMAPFINT = wxNewId();
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
	wxFlexGridSizer* FlexGridSizer3;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;

	Create(parent, wxID_ANY, _("Welcome to..."), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("wxID_ANY"));
	FlexGridSizer1 = new wxFlexGridSizer(3, 1, 0, 0);
	FlexGridSizer1->AddGrowableCol(0);
	FlexGridSizer1->AddGrowableRow(0);
	FlexGridSizer2 = new wxFlexGridSizer(1, 2, 0, 0);
	FlexGridSizer2->AddGrowableCol(1);
	FlexGridSizer2->AddGrowableRow(0);
	FlexGridSizer3 = new wxFlexGridSizer(2, 1, 0, 0);
	StaticBitmap1 = new wxStaticBitmap(this, ID_STATICBITMAP1, wxNullBitmap, wxDefaultPosition, wxSize(128,59), 0, _T("ID_STATICBITMAP1"));
	wxMemoryInputStream in((const unsigned char*) cesnet_logo_png, sizeof(cesnet_logo_png));

	 wxImage cesnet_(in);
	 StaticBitmap1->SetBitmap(wxBitmap(cesnet_));
	FlexGridSizer3->Add(StaticBitmap1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	BitmapFINT = new wxStaticBitmap(this, ID_BITMAPFINT, wxNullBitmap, wxDefaultPosition, wxSize(100,55), 0, _T("ID_BITMAPFINT"));
	wxMemoryInputStream in_fint((const unsigned char*) fint_logo_png, sizeof(fint_logo_png));

	wxImage fint_image(in_fint, wxBITMAP_TYPE_PNG);
	BitmapFINT->SetBitmap(wxBitmap(fint_image));
	FlexGridSizer3->Add(BitmapFINT, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer2->Add(FlexGridSizer3, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StaticText1 = new wxStaticText(this, ID_STATICTEXT1, wxEmptyString, wxDefaultPosition, wxSize(198,97), 0, _T("ID_STATICTEXT1"));
	StaticText1->SetLabel(wxT("FlashNET is a remote preview tool jointly developed by CESNET and FINT."));
	wxSize Size = StaticText1->GetBestSize();

	StaticText1->Wrap(StaticText1->GetSize().GetWidth());
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
}

About::~About()
{
	//(*Destroy(About)
	//*)
}

