// One-time QuPath setup for headless use: store the QuPath user directory (where the extensions folder lives).
// Headless QuPath loads extensions only from this directory, and the preference is empty until the GUI has been
// opened once. args: [0] user directory, e.g. C:/Users/<you>/QuPath/v0.7
qupath.lib.gui.prefs.PathPrefs.userPathProperty().set(args[0])
println("RNS_QUPATH_USER_DIR " + qupath.lib.gui.prefs.PathPrefs.userPathProperty().get())
