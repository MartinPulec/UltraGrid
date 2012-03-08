#ifndef SETTINGS_H
#define SETTINGS_H

#include <map>
#include <string>

class Settings
{
    public:
        Settings();
        virtual ~Settings();
        std::string GetValue(std::string key, std::string defVal = "");
        void SetValue(std::string key, std::string value);
    protected:
    private:
        void save();
        void load();
        std::map<std::string, std::string> settings;
        std::string filename;
};

#endif // SETTINGS_H
