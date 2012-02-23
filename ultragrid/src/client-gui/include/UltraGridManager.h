#ifndef ULTRAGRIDMANAGER_H
#define ULTRAGRIDMANAGER_H

#include <wx/process.h>
#include <wx/event.h>

class client_guiFrame;

class UltraGridManager
{
    public:
        UltraGridManager(client_guiFrame *parent);
        virtual ~UltraGridManager();
        void newWindow();

        void HandleTerminate(wxEvent & e);
        bool StopRunning();
    protected:
    private:
        long pid;
        wxProcess *process;
        wxEvtHandler evHandler;
        client_guiFrame *parent;
};

#endif // ULTRAGRIDMANAGER_H
