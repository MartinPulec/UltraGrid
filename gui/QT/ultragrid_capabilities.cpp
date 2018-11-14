#include "ultragrid_capabilities.hpp"

#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>

using std::list;
using std::pair;
using std::string;

UltraGridCapabilities::UltraGridCapabilities(string const &ug_path)
{
	QProcess process;

	process.start(QString::fromStdString(ug_path) + " --capabilities");

	process.waitForFinished();
	QString output = QString(process.readAllStandardOutput());
#ifdef WIN32
	QString lineSeparator = "\r\n";
#else
	QString lineSeparator = "\n";
#endif
	QStringList lines = output.split(lineSeparator);

	foreach ( const QString &line, lines ) {
		if (!line.startsWith("[capability][capture][v1]")) { // currently only capturers
			continue;
		}
		QString trimmed = line.mid(strlen("[capability][capture][v1]"));
		capturers.append(trimmed);
	}
}

UltraGridCapabilities &UltraGridCapabilities::getInstance(string const &ug_path)
{
        static UltraGridCapabilities instance{ug_path};
        return instance;
}

/// UltraGridCapabilities::capturers are in format:
/// {"id": "testcard", "name": "Testing signal", "modes": {"1920:1080:25:UYVY:i": "1080@50i"}}
/// @return list of pairs in format (id, name) - it is eg. "decklink:device=0"
list<pair<string,string>> UltraGridCapabilities::getUltraGridCapturers()
{
        list<pair<string,string>> out;
	foreach ( const QString &line, capturers ) {
                // skip v4l2 and screen that is already handled by GUI manually
                if (line.contains("v4l2") || line.contains("screen")) {
                        continue;
                }
                auto jDoc = QJsonDocument::fromJson(line.toUtf8().constData());
                auto jObj = jDoc.object();
                if (jObj.find("device") == jObj.end() || jObj.find("name") == jObj.end()) {
                        fprintf(stderr, "Corrupted JSON! Missing 'device' or 'name' node.\n");
                        continue;
                }
                out.push_back({jObj.find("device").value().toString().toUtf8().constData(),
                                jObj.find("name").value().toString().toUtf8().constData()});
        }
        return out;
}

list<pair<string,string>> UltraGridCapabilities::getCaptureModes(const std::string& device)
{
        list<pair<string,string>> ret;
	foreach ( const QString &line, capturers ) {
                auto jDoc = QJsonDocument::fromJson(line.toUtf8().constData());
                auto jObj = jDoc.object();

                if (jObj.find("device").value().toString() != QString::fromStdString(device)) {
                        continue;
                }

                auto modes = jObj.find("modes").value().toObject();
                if (jObj.find("modes") == jObj.end()) {
                        fprintf(stderr, "Corrupted JSON! Missing node 'modes'.\n");
                        continue;
                }
                for (auto it = modes.begin(); it != modes.end(); ++it) {
                        ret.push_back({it.key().toUtf8().constData(),
                                        it.value().toString().toUtf8().constData()});
                }
                break; // we do not need to iterate any further
        }
        return ret;
}

