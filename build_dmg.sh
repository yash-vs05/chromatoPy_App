#!/bin/zsh
set -e

APP_NAME="chromatoPy-desktop"
VOL_NAME="chromatoPy"
DIST_DIR="build/dist"
DMG_ROOT="build/dmg-root"
DMG_PATH="$DIST_DIR/chromatoPy.dmg"

rm -rf "$DMG_ROOT"
mkdir -p "$DMG_ROOT"

cp -R "$DIST_DIR/$APP_NAME.app" "$DMG_ROOT/"
ln -s /Applications "$DMG_ROOT/Applications"

hdiutil create \
  -volname "$VOL_NAME" \
  -srcfolder "$DMG_ROOT" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

du -sh "$DMG_PATH"
