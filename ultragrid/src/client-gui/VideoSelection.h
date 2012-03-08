#ifndef VIDEOSELECTION_H
#define VIDEOSELECTION_H

//(*Headers(VideoSelection)
#include <wx/dialog.h>
#include <wx/sizer.h>
#include <wx/button.h>
#include <wx/listbox.h>
//*)

#include "include/Settings.h"
#include "include/VideoEntry.h"

class VideoSelection: public wxDialog
{
	public:

		VideoSelection(wxWindow* parent, Settings *s, wxWindowID id=wxID_ANY,const wxPoint& pos=wxDefaultPosition,const wxSize& size=wxDefaultSize);
		virtual ~VideoSelection();

		//(*Declarations(VideoSelection)
		wxListBox* ListBox1;
		wxButton* GetList;
		//*)

		ArrayOfVideoEntries GetSelectedVideo();



	protected:

		//(*Identifiers(VideoSelection)
		static const long ID_LISTBOX1;
		static const long ID_BUTTON1;
		//*)

	private:

		//(*Handlers(VideoSelection)
		void OnInit(wxInitDialogEvent& event);
		void OnListBox1Select(wxCommandEvent& event);
		void OnListBox1Select1(wxCommandEvent& event);
		void OnGetListClick(wxCommandEvent& event);
		//*)
		void Parse(char *body, int body_len);

        ArrayOfVideoEntries videos;
		Settings *settings;

		DECLARE_EVENT_TABLE()
};

#endif
