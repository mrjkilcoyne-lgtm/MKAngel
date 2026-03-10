[app]
title = MKAngel
package.name = mkangel
package.domain = com.mrjkilcoyne
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
version = 0.2.0

# Entry point — buildozer requires main.py
# The CI workflow copies main_android.py -> main.py before building

# Python 3 includes sqlite3 built-in; no separate recipe needed
requirements = python3,kivy

# Android settings
android.permissions = INTERNET
android.api = 33
android.minapi = 24
android.ndk_api = 24
android.arch = arm64-v8a

# Orientation
orientation = portrait

# Fullscreen
fullscreen = 0

# Icon & presplash (add your own later)
# icon.filename = %(source.dir)s/assets/icon.png
# presplash.filename = %(source.dir)s/assets/splash.png

# Include the glm and app packages
source.include_patterns = glm/**/*.py,app/**/*.py,main_android.py
source.exclude_dirs = .github,tests,bin,.buildozer

# iOS (future)
ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master

[buildozer]
log_level = 2
warn_on_root = 1
