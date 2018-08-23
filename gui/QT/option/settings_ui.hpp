#ifndef SETTINGS_UI_HPP
#define SETTINGS_UI_HPP

#include <QObject>
#include <QString>

#include "available_settings.hpp"
#include "ui_ultragrid_window.h"
#include "ui_settings.h"
#include "settings.hpp"

class SettingsUi : public QObject{
	Q_OBJECT

public:
	void init(Settings *settings, AvailableSettings *availableSettings);
	void initMainWin(Ui::UltragridWindow *ui);
	void initSettingsWin(Ui::Settings *ui);

private:
	Ui::UltragridWindow *mainWin = nullptr;
	Ui::Settings *settingsWin = nullptr;
	Settings *settings = nullptr;
	AvailableSettings *availableSettings = nullptr;


	void populateComboBox(QComboBox *box,
			SettingType type,
			const std::vector<std::string> &whitelist = {});

	void initVideoCompress();
	void initVideoSource();
	void initVideoDisplay();

	void initAudioSource();
	void initAudioPlayback();
	void initAudioCompression();

	void videoCompressionCallback(Option &opt); 
	void videoSourceCallback(Option &opt, bool suboption); 
	void videoDisplayCallback(Option &opt); 

	void audioSourceCallback(Option &opt);
	void audioPlaybackCallback(Option &opt);
	void audioCompressionCallback(Option &opt, bool suboption);

	bool isAdvancedMode();

	void setComboBox(QComboBox *box, const std::string &opt, int idx);
	void setString(const std::string &opt, const QString &str);

private slots:
	void setAdvanced(bool enable);
	void setVideoCompression(int idx);
	void setVideoBitrate(const QString &);
	void setVideoSourceMode(int idx);

	void setHwAccel(bool b);

	void test();

signals:
	void changed();
};


#endif //SETTINGS_UI
