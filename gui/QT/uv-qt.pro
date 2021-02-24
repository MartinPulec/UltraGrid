######################################################################
# Automatically generated by qmake (3.0) Wed Jun 21 13:05:24 2017
######################################################################

QMAKE_TARGET_BUNDLE_PREFIX = cz.cesnet.ultragrid
TEMPLATE = app
TARGET = uv-qt
INCLUDEPATH += .
INCLUDEPATH += $$PWD/../../tools/
INCLUDEPATH += $$PWD/../../src
INCLUDEPATH += window/
INCLUDEPATH += util/
INCLUDEPATH += widget/
INCLUDEPATH += option/
RC_FILE = uv-qt.rc

DEFINES += GUI_BUILD

QT += widgets

CONFIG += c++11

LIBS += $$PWD/../../tools/astat.a
macx {
	LIBS += -framework CoreFoundation
} win32 {
	LIBS += -lWs2_32
}

astat.target = astat_lib
astat.commands = cd $$PWD/../../tools && make -f Makefile.astat lib

QMAKE_EXTRA_TARGETS += astat
PRE_TARGETDEPS += astat_lib

system("which git"): HAS_GIT = TRUE

equals(HAS_GIT, "TRUE") {
	DEFINES += GIT_CURRENT_SHA1="\\\"$(shell git -C \""$$_PRO_FILE_PWD_"\" rev-parse --short HEAD)\\\""
	DEFINES += GIT_CURRENT_BRANCH="\\\"$(shell git -C \""$$_PRO_FILE_PWD_"\" name-rev --name-only HEAD)\\\""
}


# Input
HEADERS += window/ultragrid_window.hpp \
	option/available_settings.hpp \
	option/settings.hpp \
	option/settings_ui.hpp \
	widget/previewWidget.hpp \
	window/log_window.hpp \
	../../tools/astat.h \
	../../src/shared_mem_frame.hpp \
	widget/vuMeterWidget.hpp \
	window/settings_window.hpp \
	option/widget_ui.hpp \
	option/checkable_ui.hpp \
	option/checkbox_ui.hpp \
	option/textOpt_ui.hpp \
	option/actionCheckable_ui.hpp \
	option/lineedit_ui.hpp \
	option/spinbox_ui.hpp \
	option/combobox_ui.hpp \
	option/groupBox_ui.hpp \
	option/radioButton_ui.hpp \
	option/audio_opts.hpp \
	option/video_opts.hpp \
	util/overload.hpp \

FORMS += ui/ultragrid_window.ui \
	ui/log_window.ui \
	ui/settings.ui

SOURCES += window/ultragrid_window.cpp \
	option/available_settings.cpp \
	option/settings.cpp \
	option/settings_ui.cpp \
	widget/previewWidget.cpp \
	window/log_window.cpp \
	widget/vuMeterWidget.cpp \
	window/settings_window.cpp \
	option/widget_ui.cpp \
	option/checkable_ui.cpp \
	option/checkbox_ui.cpp \
	option/actionCheckable_ui.cpp \
	option/textOpt_ui.cpp \
	option/lineedit_ui.cpp \
	option/spinbox_ui.cpp \
	option/combobox_ui.cpp \
	option/groupBox_ui.cpp \
	option/radioButton_ui.cpp \
	option/audio_opts.cpp \
	option/video_opts.cpp \
	../../src/shared_mem_frame.cpp \
	main.cpp
