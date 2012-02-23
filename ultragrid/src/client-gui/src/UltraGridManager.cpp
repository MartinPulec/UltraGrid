#include "../include/UltraGridManager.h"
#include <stdexcept>
#include <iostream>
#include <assert.h>

#include "../client_guiMain.h"

UltraGridManager::UltraGridManager(client_guiFrame *p) :
    process(0), parent(p)
{
    evHandler.Connect(wxEVT_END_PROCESS, (wxObjectEventFunction) &UltraGridManager::HandleTerminate, NULL, (wxEvtHandler*) this);
    assert(p != 0);
}

UltraGridManager::~UltraGridManager()
{
    //dtor
}

void UltraGridManager::newWindow()
{
    if(process) {
        return;
    }

    process = new wxProcess(&evHandler);
    pid = wxExecute(wxT("uv -d sdl"), wxEXEC_ASYNC, this->process);
    if(!pid)
        throw std::runtime_error(std::string("Failed to create process"));
}

void UltraGridManager::HandleTerminate(wxEvent & e)
{
    delete process;
    process = 0;
    pid = 0;
    parent->NotifyWindowClosed();
}

bool UltraGridManager::StopRunning()
{
    if(process) {
        wxProcess::Kill(pid, wxSIGTERM);
        return true;
    }

    return false;
}
