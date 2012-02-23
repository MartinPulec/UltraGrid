#include "../include/Settings.h"
#include <fstream>
#include <string.h>
#include <stdlib.h>

using namespace std;

Settings::Settings()
{
    const char *home_dir = getenv("HOME");
    filename = string("") + home_dir + "/" + ".ugcrc";
    load();
}

Settings::~Settings()
{
    //dtor
}

std::string Settings::GetValue(std::string key)
{
    map<string, string>::iterator it = settings.find(key);
    if (it == settings.end())
        return string();
    else
        return (*it).second;
}

void Settings::SetValue(std::string key, std::string value)
{
    settings[key] = value;
    save();
}

void Settings::save()
{
        map<string, string>::iterator it = settings.begin();
        ofstream ofs (filename.c_str(), ios_base::out );
        for ( ; it != settings.end(); ++it) {
            ofs << (*it).first << "=" << (*it).second << std::endl;
        }
        ofs.close();
}

void Settings::load()
{
    ifstream ifs (filename.c_str(), ifstream::in );
    if(!ifs)
        return;
    else {
        char buf[1024];
        char *save_ptr = NULL;
        string key, value;
        while(ifs.getline(buf, 1024)) {
            key = strtok_r(buf, "=", &save_ptr);
            value = save_ptr;
            if(!key.empty())
                settings.insert(pair<string, string>(key, value));
        }
        ifs.close();
    }
}
