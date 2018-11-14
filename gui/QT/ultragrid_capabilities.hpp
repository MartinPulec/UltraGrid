#ifndef ULTRAGRID_CAPABILITIES_HPP
#define ULTRAGRID_CAPABILITIES_HPP

#include <list>
#include <QStringList>
#include <utility>
#include <string>

struct UltraGridCapabilities {
        static UltraGridCapabilities &getInstance(std::string const &ug_path);
        UltraGridCapabilities(UltraGridCapabilities const&) = delete;
        UltraGridCapabilities operator=(UltraGridCapabilities const&) = delete;

        std::list<std::pair<std::string, std::string>> getUltraGridCapturers();
        std::list<std::pair<std::string, std::string>> getCaptureModes(const std::string& device);

private:
        UltraGridCapabilities(std::string const &ug_path);
        QStringList capturers;
};

#endif // ULTRAGRID_CAPABILITIES_HPP
