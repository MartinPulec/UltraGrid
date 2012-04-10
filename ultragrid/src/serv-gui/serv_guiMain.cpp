/***************************************************************
 * Name:      serv_guiMain.cpp
 * Purpose:   Code for Application Frame
 * Author:    Martin Pulec (pulec@cesnet.cz)
 * Created:   2012-02-20
 * Copyright: Martin Pulec ()
 * License:
 **************************************************************/

#include "serv_guiMain.h"
#include <wx/msgdlg.h>

//(*InternalHeaders(serv_guiFrame)
#include <wx/string.h>
#include <wx/intl.h>
//*)

#include <wx/textfile.h>

#include <sys/types.h>          /* See NOTES */
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netdb.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>

#include <glob.h>

#include <algorithm>
#include <string>
#include <stdexcept>

#include "VideoProperties.h"

#include "../client-gui/About.h"
#include "../client-gui/include/Utils.h"

//helper functions
enum wxbuildinfoformat {
    short_f, long_f };

wxString wxbuildinfo(wxbuildinfoformat format)
{
    wxString wxbuild(wxVERSION_STRING);

    if (format == long_f )
    {
#if defined(__WXMSW__)
        wxbuild << _T("-Windows");
#elif defined(__UNIX__)
        wxbuild << _T("-Linux");
#endif

#if wxUSE_UNICODE
        wxbuild << _T("-Unicode build");
#else
        wxbuild << _T("-ANSI build");
#endif // wxUSE_UNICODE
    }

    return wxbuild;
}

//(*IdInit(serv_guiFrame)
const long serv_guiFrame::ID_STATICTEXT1 = wxNewId();
const long serv_guiFrame::ID_CHECKLISTBOX1 = wxNewId();
const long serv_guiFrame::ID_BUTTON1 = wxNewId();
const long serv_guiFrame::ID_BUTTON2 = wxNewId();
const long serv_guiFrame::ID_Edit = wxNewId();
const long serv_guiFrame::ID_BUTTON3 = wxNewId();
const long serv_guiFrame::idMenuQuit = wxNewId();
const long serv_guiFrame::idMenuAbout = wxNewId();
const long serv_guiFrame::ID_STATUSBAR1 = wxNewId();
//*)

BEGIN_EVENT_TABLE(serv_guiFrame,wxFrame)
    //(*EventTable(serv_guiFrame)
    //*)
END_EVENT_TABLE()

serv_guiFrame::serv_guiFrame(wxWindow* parent,wxWindowID id)
{
    //(*Initialize(serv_guiFrame)
    wxMenuItem* MenuItem2;
    wxMenuItem* MenuItem1;
    wxFlexGridSizer* FlexGridSizer1;
    wxMenu* Menu1;
    wxBoxSizer* BoxSizer1;
    wxMenuBar* MenuBar1;
    wxMenu* Menu2;

    Create(parent, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_FRAME_STYLE, _T("wxID_ANY"));
    FlexGridSizer1 = new wxFlexGridSizer(3, 1, 0, 0);
    FlexGridSizer1->AddGrowableCol(0);
    FlexGridSizer1->AddGrowableCol(1);
    FlexGridSizer1->AddGrowableCol(2);
    FlexGridSizer1->AddGrowableRow(1);
    StaticText1 = new wxStaticText(this, ID_STATICTEXT1, _("Provided stream locations:"), wxDefaultPosition, wxDefaultSize, 0, _T("ID_STATICTEXT1"));
    FlexGridSizer1->Add(StaticText1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    CheckListBox1 = new wxCheckListBox(this, ID_CHECKLISTBOX1, wxDefaultPosition, wxSize(580,87), 0, 0, 0, wxDefaultValidator, _T("ID_CHECKLISTBOX1"));
    CheckListBox1->SetMaxSize(wxDLG_UNIT(this,wxSize(-1,-1)));
    FlexGridSizer1->Add(CheckListBox1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    BoxSizer1 = new wxBoxSizer(wxHORIZONTAL);
    Button1 = new wxButton(this, ID_BUTTON1, _("Add New Path"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON1"));
    BoxSizer1->Add(Button1, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Button2 = new wxButton(this, ID_BUTTON2, _("Remove Selected"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON2"));
    BoxSizer1->Add(Button2, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Edit = new wxButton(this, ID_Edit, _("Edit Properties"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_Edit"));
    BoxSizer1->Add(Edit, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    BoxSizer1->Add(-1,-1,1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    Button3 = new wxButton(this, ID_BUTTON3, _("Save"), wxDefaultPosition, wxDefaultSize, 0, wxDefaultValidator, _T("ID_BUTTON3"));
    BoxSizer1->Add(Button3, 1, wxALL|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    FlexGridSizer1->Add(BoxSizer1, 1, wxALL|wxEXPAND|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL, 5);
    SetSizer(FlexGridSizer1);
    MenuBar1 = new wxMenuBar();
    Menu1 = new wxMenu();
    MenuItem1 = new wxMenuItem(Menu1, idMenuQuit, _("Quit\tAlt-F4"), _("Quit the application"), wxITEM_NORMAL);
    Menu1->Append(MenuItem1);
    MenuBar1->Append(Menu1, _("&File"));
    Menu2 = new wxMenu();
    MenuItem2 = new wxMenuItem(Menu2, idMenuAbout, _("About\tF1"), _("Show info about this application"), wxITEM_NORMAL);
    Menu2->Append(MenuItem2);
    MenuBar1->Append(Menu2, _("Help"));
    SetMenuBar(MenuBar1);
    StatusBar1 = new wxStatusBar(this, ID_STATUSBAR1, 0, _T("ID_STATUSBAR1"));
    int __wxStatusBarWidths_1[1] = { -1 };
    int __wxStatusBarStyles_1[1] = { wxSB_NORMAL };
    StatusBar1->SetFieldsCount(1,__wxStatusBarWidths_1);
    StatusBar1->SetStatusStyles(1,__wxStatusBarStyles_1);
    SetStatusBar(StatusBar1);
    PathSelection = new wxDirDialog(this, _("Select directory"), wxEmptyString, wxDD_DEFAULT_STYLE, wxDefaultPosition, wxDefaultSize, _T("wxDirDialog"));
    FlexGridSizer1->Fit(this);
    FlexGridSizer1->SetSizeHints(this);

    Connect(ID_CHECKLISTBOX1,wxEVT_COMMAND_CHECKLISTBOX_TOGGLED,(wxObjectEventFunction)&serv_guiFrame::OnCheckListBox1Toggled);
    Connect(ID_BUTTON1,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&serv_guiFrame::OnButton1Click);
    Connect(ID_BUTTON2,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&serv_guiFrame::OnButton2Click);
    Connect(ID_Edit,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&serv_guiFrame::OnButton4Click);
    Connect(ID_BUTTON3,wxEVT_COMMAND_BUTTON_CLICKED,(wxObjectEventFunction)&serv_guiFrame::OnButton3Click);
    Connect(idMenuQuit,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&serv_guiFrame::OnQuit);
    Connect(idMenuAbout,wxEVT_COMMAND_MENU_SELECTED,(wxObjectEventFunction)&serv_guiFrame::OnAbout);
    //*)

    const char *home_dir = getenv("HOME");
    wxString path = wxString(home_dir, wxConvUTF8) + wxString("/", wxConvUTF8) + wxString("/.ugsrc/", wxConvUTF8);
    filename = path + wxString("index.lst", wxConvUTF8);
    mkdir(path.mb_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    open(filename.mb_str(), O_CREAT | O_WRONLY, 0666);
    LoadSettings();
}

serv_guiFrame::~serv_guiFrame()
{
    //(*Destroy(serv_guiFrame)
    //*)
}

void serv_guiFrame::OnQuit(wxCommandEvent& event)
{
    Close();
}

void serv_guiFrame::OnAbout(wxCommandEvent& event)
{
    About dlg(this);

    dlg.ShowModal();
/*    wxString msg = wxbuildinfo(long_f);
    wxMessageBox(msg, _("Welcome to..."));*/
}

void serv_guiFrame::OnCheckListBox1Toggled(wxCommandEvent& event)
{
}

void GetPrimaryIp(char* buffer, size_t buflen)
{
    assert(buflen >= 16);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    assert(sock != -1);

    const char* kGoogleDnsIp = "8.8.8.8";
    uint16_t kDnsPort = 53;
    struct sockaddr_in serv;
    memset(&serv, 0, sizeof(serv));
    serv.sin_family = AF_INET;
    serv.sin_addr.s_addr = inet_addr(kGoogleDnsIp);
    serv.sin_port = htons(kDnsPort);

    int err = connect(sock, (const sockaddr*) &serv, sizeof(serv));
    assert(err != -1);

    sockaddr_in name;
    socklen_t namelen = sizeof(name);
    err = getsockname(sock, (sockaddr*) &name, &namelen);
    assert(err != -1);

    const char* p = inet_ntop(AF_INET, &name.sin_addr, buffer, buflen);
    assert(p);

    close(sock);
}

bool GetNameToIp(char *ip, char *name, int len)
{
    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
    inet_aton(ip, &sin.sin_addr);

    if (getnameinfo((struct sockaddr *) &sin, sizeof(struct sockaddr_in), name, len, NULL,
            0, 0 /*NI_NUMERICHOST | NI_NUMERICSERV*/) == 0)
        return true;
    else
        return false;
}

void serv_guiFrame::OnButton1Click(wxCommandEvent& event)
{
    int res;
    char buf[1024];
    char hostname[NI_MAXHOST];
    res = PathSelection->ShowModal();
    if(res == wxID_OK) {
        try {
            VideoEntry newEntry;

            GetPrimaryIp(buf, 1024);
            GetNameToIp(buf, hostname, NI_MAXHOST);

            wxString path = wxString("rtp://", wxConvUTF8) + wxString(hostname, wxConvUTF8) + PathSelection->GetPath();

            newEntry.URL = path;

            this->ScanDirectory(PathSelection->GetPath(), newEntry);

            this->videos.Add(newEntry);
            CheckListBox1->InsertItems(1u, &path, CheckListBox1->GetCount());
        } catch (std::exception &e) {
            wxString message = wxString(e.what(), wxConvUTF8);
            wxMessageBox(message, wxT("Error"), wxICON_ERROR);
        }
    }
    StatusBar1->PushStatusText(wxT("unsaved changes"));
}

void serv_guiFrame::OnButton2Click(wxCommandEvent& event)
{
    unsigned int i = 0;
    while(i < CheckListBox1->GetCount()) {
        if(CheckListBox1->IsChecked(i)) {
            CheckListBox1->Delete(i);
            this->videos.RemoveAt(i);
        } else {
            i++;
        }
    }
    StatusBar1->PushStatusText(wxT("unsaved changes"));
}

void serv_guiFrame::OnButton3Click(wxCommandEvent& event)
{
    wxTextFile settings(filename);
    settings.Clear();
    unsigned int i;
    for(i = 0u; i < this->videos.GetCount(); ++i) {
        settings.AddLine(this->videos[i].Serialize());
    }

    settings.Write();
    settings.Close();
    StatusBar1->PushStatusText(wxT("saved"));
}

void serv_guiFrame::LoadSettings(void)
{
    wxTextFile settings(filename);
    wxString str;

    settings.Open();

    for ( str = settings.GetFirstLine(); !settings.Eof(); str = settings.GetNextLine() )
    {
        VideoEntry newVideo(str);
        this->videos.Add(newVideo);
        CheckListBox1->InsertItems(1u, &newVideo.URL, CheckListBox1->GetCount());
    }

    settings.Close();
}

void serv_guiFrame::OnButton4Click(wxCommandEvent& event)
{
    for (int i = 0; i < CheckListBox1->GetCount(); ++i) {
        if(CheckListBox1->IsChecked(i)) {
            VideoProperties dlg(this);
            dlg.FPSCtrl->SetValue(Utils::FromCDouble(this->videos[i].fps, 2));
            dlg.Name->SetLabel(this->videos[i].URL);

            if(dlg.ShowModal() == wxID_OK) {
                wxString number = dlg.FPSCtrl->GetValue();
                if(!number.ToDouble(&this->videos[i].fps)){ /* error! */ }
            }
        }
    }

    StatusBar1->PushStatusText(wxT("unsaved changes"));
}

void serv_guiFrame::ScanDirectory(wxString path, VideoEntry &entry)
{
    glob_t dir_listening;

    entry.format = wxT("none");

    for (int i = 0; i < possibleFileFormatsCount; ++i) {
        std::string extension(possibleFileFormats[i]);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        wxString glob_pattern = path + wxT("/*.") + wxString(extension.c_str(), wxConvUTF8);

        int ret = glob(glob_pattern.mb_str(), 0, NULL, &dir_listening);
        if (ret) {
            continue;
        } else  {
            entry.format = wxString(possibleFileFormats[i], wxConvUTF8);
            entry.total_frames = dir_listening.gl_pathc;
            break;
        }
    }

    if(entry.format == wxT("none")) {
        throw std::runtime_error("No image/video files found in directory");
    }
}
