#include "../include/VideoEntry.h"
#include "../include/Utils.h"

using namespace std;

#include <iostream>
#include <wx/tokenzr.h>
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!

WX_DEFINE_OBJARRAY(ArrayOfVideoEntries);

const char *possibleFileFormats[] = {"DPX", "EXR", "TIFF"};
int possibleFileFormatsCount = (sizeof(possibleFileFormats) / sizeof(const char *));

VideoEntry::VideoEntry() :
    URL(L""),
    format(L"none"),
    fps(25.0),
    total_frames(0),
    colorSpace(L"file")
{
    //ctor
}

VideoEntry::VideoEntry(wxString & encoded)
{
    wxStringTokenizer tkz_words(encoded, wxT(" "));
    this->URL = tkz_words.GetNextToken();
    wxString fpsStr = tkz_words.GetNextToken();
    wxString totalFramesStr = tkz_words.GetNextToken();
    this->format = tkz_words.GetNextToken();
    this->colorSpace = tkz_words.GetNextToken();

    double val_fps;
    long val_total_frames;
    fpsStr.ToDouble(&val_fps);
    this->fps = val_fps;
    totalFramesStr.ToLong(&val_total_frames);
    this->total_frames = val_total_frames;

    this->audioFile = tkz_words.GetNextToken();
}

VideoEntry::~VideoEntry()
{
    //dtor
}

wxString VideoEntry::Serialize()
{
    wxString line;

    line << URL << wxT(" ") << Utils::FromCDouble(fps, 2) << wxT(" ") << total_frames <<  wxT(" ") << format << wxT(" ") << colorSpace << wxT(" ") << audioFile;

    return line;
}


