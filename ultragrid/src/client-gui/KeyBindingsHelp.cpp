#include "KeyBindingsHelp.h"

//(*InternalHeaders(KeyBindingsHelp)
#include <wx/button.h>
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(KeyBindingsHelp)
const long KeyBindingsHelp::ID_HTMLWINDOW1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(KeyBindingsHelp,wxDialog)
	//(*EventTable(KeyBindingsHelp)
	//*)
END_EVENT_TABLE()

KeyBindingsHelp::KeyBindingsHelp(wxWindow* parent,wxWindowID id,const wxPoint& pos,const wxSize& size)
{
	//(*Initialize(KeyBindingsHelp)
	wxFlexGridSizer* FlexGridSizer1;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;
	
	Create(parent, id, _("Keybindings"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(371,308));
	Move(wxDefaultPosition);
	FlexGridSizer1 = new wxFlexGridSizer(0, 1, 0, 0);
	FlexGridSizer1->AddGrowableCol(0);
	FlexGridSizer1->AddGrowableRow(0);
	HtmlWindow1 = new wxHtmlWindow(this, ID_HTMLWINDOW1, wxDefaultPosition, wxSize(426,328), wxHW_SCROLLBAR_AUTO, _T("ID_HTMLWINDOW1"));
	HtmlWindow1->SetPage(_("<h5>Available keybindings:</h5>\n<dl>\n<dt>F</dt>\n<dt>&lt;Alt&gt; - Enter</dt>\n<dd>toggle fullscreen</dd>\n<dt>&lt;space&gt;</dt>\n<dd>toggle pause</dd>\n<dt>K</dt>\n<dd>pause</dd>\n<dt>J</dt>\n<dd>previous frame</dd>\n<dt>L</dt>\n<dd>following frame</dd>\n<dt>C</dt>\n<dd>toggle lightness (color / mono / greyscale)</dd>\n<dt>R</dt>\n<dt>G</dt>\n<dt>B</dt>\n<dd>show only red / green / blue channel</dd>\n<dt>&lt;Ctrl&gt;/&lt;Cmd&gt;-R</dt>\n<dt>&lt;Ctrl&gt;/&lt;Cmd&gt;-G</dt>\n<dt>&lt;Ctrl&gt;/&lt;Cmd&gt;-B</dt>\n<dd>hide red / green / blue channel (use Ctrl in Linux, cmd in Mac OS X)<dd>\n<dt>P;</dt>\n<dd>toggle loop</dd>\n<dt>+</dt>\n<dd>zoom in</dd>\n<dt>-</dt>\n<dd>zoom out</dd>\n<dt>&larr;</dt>\n<dt>&uarr;</dt>\n<dt>&rarr;</dt>\n<dt>&darr;</dt>\n<dd>move viewport left/up/right/down</dd>\n</dl>"));
	FlexGridSizer1->Add(HtmlWindow1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StdDialogButtonSizer1 = new wxStdDialogButtonSizer();
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_OK, wxEmptyString));
	StdDialogButtonSizer1->Realize();
	FlexGridSizer1->Add(StdDialogButtonSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	SetSizer(FlexGridSizer1);
	FlexGridSizer1->SetSizeHints(this);
	//*)
}

KeyBindingsHelp::~KeyBindingsHelp()
{
	//(*Destroy(KeyBindingsHelp)
	//*)
}

