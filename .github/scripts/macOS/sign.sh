#!/bin/sh -eu
##
## Signs given application bundle
##
## Usage:
##  sign.sh <app_bundle_directory>
##
## Environment variables:
## - **KEY_CHAIN**              - keychain containing the CESNET signing key
## - **KEY_CHAIN_PASS**         - password to the above keychain
## - **notarytool_credentials** - developer credentials to be used with notarytool (in format user:password:team_id)
##
## KEY_CHAIN and KEY_CHAIN_PASS are set by .github/scripts/environment.sh
## if apple_key_p12_b64 GH environment workflow is defined.

# The command-line parameters are not used by the CI but are intended
# to be used interactively for debugging.
if [ $# -eq 1 ] && { [ "$1" = -h ] || [ "$1" = --help ] ||
                [ "$1" = help ]; }; then
        printf "Usage:\n"
        printf "\t%s [--sign-only] <bundle_path>\n" "$0"
        printf "\nSigns and notarizes the application bundle.\n"
        printf "\nUse \"--sign-only\" to skip application notarization.\n"
        exit 0
fi
sign_only=
if [ "${1-}" = --sign-only ]; then
        sign_only=1
        shift
fi

set -x

APP=${1?Appname must be passed as a first argument}

if [ -z "${KEY_CHAIN-}" ] || [ -z "${KEY_CHAIN_PASS-}" ]; then
        echo "Could not find key to sign the application" 2>&1
        exit 1
fi
if  [ -z "${notarytool_credentials-}" ] && [ ! $sign_only ]; then
        echo "Could not find notarytool credentials" 2>&1
        exit 1
fi

security unlock-keychain -p "$KEY_CHAIN_PASS" "$KEY_CHAIN"

# Sign the application
# Libs need to be signed explicitly for some reason
for f in $(find "$APP/Contents/libs" -type f) $APP; do
        codesign --force --deep -s CESNET --options runtime --entitlements data/entitlements.mac.plist -v "$f"
done
#codesign --force --deep -s CESNET --options runtime -v $APP/Contents/MacOS/uv-qt

if [ $sign_only ]; then
        exit 0
fi

# attempt to "report" crashing notarytool on Bus Error, eg. here:
# https://github.com/MartinPulec/UltraGrid/actions/runs/13280267692/job/37077042169
# TODO TOREMOVE later - 1. this worked exactly as thought, report was sent
# and apple repair it or 2. it doesn't work or Apple ignores that
sudo defaults write /Library/Application\ Support/CrashReporter/\
DiagnosticMessagesHistory.plist AutoSubmit -bool true
sudo defaults write /Library/Application\ Support/CrashReporter/\
DiagnosticMessagesHistory.plist ThirdPartyDataSubmit -bool true
sudo launchctl kickstart -k system/com.apple.ReportCrash.Root
sudo launchctl kickstart -k system/com.apple.SubmitDiagInfo

# Zip and send for notarization
ZIP_FILE=uv-qt.zip
ditto -c -k --keepParent "$APP" $ZIP_FILE
set +x
DEVELOPER_USERNAME=$(echo "$notarytool_credentials" | cut -d: -f1)
DEVELOPER_PASSWORD=$(echo "$notarytool_credentials" | cut -d: -f2)
DEVELOPER_TEAMID=$(echo "$notarytool_credentials" | cut -d: -f3)
xcrun notarytool submit $ZIP_FILE --apple-id "$DEVELOPER_USERNAME" --team-id "$DEVELOPER_TEAMID" --password "$DEVELOPER_PASSWORD" --wait
set -x
# If everything is ok, staple the app
xcrun stapler staple "$APP"

