#ifndef COMPRESSIONSETTING_H
#define COMPRESSIONSETTING_H

//(*Headers(CompressionSetting)
#include <wx/spinctrl.h>
#include <wx/dialog.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/choice.h>
//*)

#include <wx/hashmap.h>

class CompressionSetting: public wxDialog
{
	public:

		CompressionSetting(wxWindow* parent);
		virtual ~CompressionSetting();

		void SetValue(wxString val);

		//(*Declarations(CompressionSetting)
		wxStaticText* StaticText1;
		wxChoice* Choice1;
		wxStaticText* description;
		wxStaticText* StaticText2;
		wxSpinCtrl* SpinCtrl1;
		//*)

	protected:

		//(*Identifiers(CompressionSetting)
		static const long ID_STATICTEXT1;
		static const long ID_CHOICE1;
		static const long ID_SPINCTRL1;
		static const long ID_STATICTEXT2;
		static const long ID_STATICTEXT3;
		//*)

	private:

		//(*Handlers(CompressionSetting)
		void OnRadioButton1Select(wxCommandEvent& event);
		void OnInit(wxInitDialogEvent& event);
		void OnChoice1Select(wxCommandEvent& event);
		//*)

		void SelectValue(int i);


		WX_DECLARE_STRING_HASH_MAP( wxString, StringHash );
		StringHash compDesc;

		wxFlexGridSizer*sizer;


		DECLARE_EVENT_TABLE()
};

#endif
