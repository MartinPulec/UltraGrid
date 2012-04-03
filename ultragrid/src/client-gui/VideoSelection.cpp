#include "VideoSelection.h"

//(*InternalHeaders(VideoSelection)
#include <wx/string.h>
#include <wx/intl.h>
//*)

#include <wx/tokenzr.h>
#include <wx/msgdlg.h>

#include <stdexcept>

#include "include/sp_client.h"

//(*IdInit(VideoSelection)
const long VideoSelection::ID_LISTBOX1 = wxNewId();
const long VideoSelection::ID_BUTTON1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(VideoSelection,wxDialog)
	//(*EventTable(VideoSelection)
	//*)
END_EVENT_TABLE()

VideoSelection::VideoSelection(wxWindow* parent, Settings *s, wxWindowID id,const wxPoint& pos,const wxSize& size):
    settings(s)
{
	//(*Initialize(VideoSelection)
	wxFlexGridSizer* FlexGridSizer1;
	wxBoxSizer* BoxSizer1;
	wxStdDialogButtonSizer* StdDialogButtonSizer1;

	Create(parent, wxID_ANY, _("Select video"), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("wxID_ANY"));
	FlexGridSizer1 = new wxFlexGridSizer(0, 1, 0, 0);
	ListBox1 = new wxListBox(this, ID_LISTBOX1, wxDefaultPosition, wxSize(527,174), 0, 0, 0, wxDefaultValidator, _T("ID_LISTBOX1"));
	FlexGridSizer1->Add(ListBox1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	BoxSizer1 = new wxBoxSizer(wxHORIZONTAL);
	GetList = new wxButton(this, ID_BUTTON1, _("Get List"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON1"));
	BoxSizer1->Add(GetList, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	BoxSizer1->Add(-1,-1,1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	StdDialogButtonSizer1 = new wxStdDialogButtonSizer();
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_OK, wxEmptyString));
	StdDialogButtonSizer1->AddButton(new wxButton(this, wxID_CANCEL, wxEmptyString));
	StdDialogButtonSizer1->Realize();
	BoxSizer1->Add(StdDialogButtonSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	FlexGridSizer1->Add(BoxSizer1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
	SetSizer(FlexGridSizer1);
	FlexGridSizer1->Fit(this);
	FlexGridSizer1->SetSizeHints(this);

	Connect(ID_LISTBOX1,wxEVT_COMMAND_LISTBOX_SELECTED,(wxObjectEventFunction)&VideoSelection::OnListBox1Select1);
	Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&VideoSelection::OnGetListClick);
	Connect(wxID_ANY,wxEVT_INIT_DIALOG,(wxObjectEventFunction)&VideoSelection::OnInit);
	//*)
}

VideoSelection::~VideoSelection()
{
	//(*Destroy(VideoSelection)
	//*)
}


void VideoSelection::OnInit(wxInitDialogEvent& event)
{
}

void VideoSelection::OnListBox1Select(wxCommandEvent& event)
{
}

void VideoSelection::OnListBox1Select1(wxCommandEvent& event)
{
}

void VideoSelection::OnGetListClick(wxCommandEvent& event)
{
    sp_client connection;

    try {
        connection.connect_to(settings->GetValue("server"), 5100);
        struct message get_req;
        struct response resp;
        get_req.msg = "GET index.lst\r\n";
        get_req.len = strlen(get_req.msg);
        connection.send(&get_req, &resp);
        if(resp.code != 200) {
            wxString msg;
            msg << resp.code << L" " << wxString::FromUTF8((resp.msg));
            wxMessageBox(msg, _("Error downloading media file"));
        }

        Parse(resp.body, resp.body_len);

    } catch (std::exception &e) {
        wxString msg = wxString::FromUTF8(e.what());
        wxMessageBox(msg, _("Error downloading media file"));
    }

    connection.disconnect();
}

void VideoSelection::Parse(char *body, int body_len)
{
    ListBox1->Clear();
    this->videos.Clear();

    wxString data = wxString::FromUTF8(body, body_len);
    wxStringTokenizer tkz(data, wxT("\r\n"));
    while ( tkz.HasMoreTokens() )
    {
        wxString token = tkz.GetNextToken();
        wxStringTokenizer tkz_words(token, wxT(" "));
        wxString URL = tkz_words.GetNextToken();
        wxString fps = tkz_words.GetNextToken();
        wxString total_frames = tkz_words.GetNextToken();
        wxString format = tkz_words.GetNextToken();
        wxString colorSpace = tkz_words.GetNextToken();

        VideoEntry newItem;

        newItem.URL = URL;
        newItem.format = format;
        double val_fps;
        long val_total_frames;
        fps.ToDouble(&val_fps);
        newItem.fps = val_fps;
        total_frames.ToLong(&val_total_frames);
        newItem.total_frames = val_total_frames;

        newItem.colorSpace = colorSpace;
        this->videos.Add(newItem);
    }

    for(int i = 0; i < videos.GetCount(); i++) {
        ListBox1->Append(videos[i].URL, (void*) &videos[i]);
    }
}

ArrayOfVideoEntries VideoSelection::GetSelectedVideo()
{
    wxArrayInt playListIndices;
    ArrayOfVideoEntries res;
    ListBox1->GetSelections(playListIndices);

    for (int i = 0; i < playListIndices.Count(); ++i) {
        res.Insert(*(VideoEntry *)(ListBox1->GetClientData(playListIndices[i])), i);
    }

    return res;
}
