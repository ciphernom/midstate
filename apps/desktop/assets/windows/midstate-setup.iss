; Midstate Desktop's Windows installer, midstate-windows-amd64-setup.exe.
;
; CI builds it from the release binaries (.github/workflows/build.yml):
;
;   ISCC /DAppVersion=6.0.2 /DBinDir=<release dir> /O<output dir> midstate-setup.iss
;
; It installs midstate-desktop.exe with midstate.exe beside it (mining runs
; midstate.exe as the pool and the hasher, and looks for it next to the app),
; adds a Start menu entry and, unless its box is unticked, a desktop
; shortcut, both with the app's icon, and registers an uninstaller under
; Settings > Apps.
;
; It installs for the current user, into %LOCALAPPDATA%\Programs\Midstate,
; so it needs no administrator rights. `/ALLUSERS` on the command line
; installs for everyone instead. An upgrade replaces the files in place:
; Windows knows the app by AppId, which must never change.
;
; The wallet is not in here. Wallets and chain data live in
; %APPDATA%\midstate-desktop, which neither setup nor the uninstaller touches.

; PNG wizard images need 6.5.2.
#if Ver < EncodeVer(6, 5, 2, 0)
  #error Inno Setup 6.5.2 or later is needed to build this installer
#endif

; Both are normally given on the command line, as above.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef BinDir
  #define BinDir "..\..\..\..\target\x86_64-pc-windows-msvc\release"
#endif

[Setup]
AppId={{011CAA24-6F61-42E2-909A-E5B4D1BF0617}
AppName=Midstate Desktop
AppVersion={#AppVersion}
AppPublisher=Midstate
AppPublisherURL=https://github.com/ciphernom/midstate
AppSupportURL=https://github.com/ciphernom/midstate/issues
AppUpdatesURL=https://github.com/ciphernom/midstate/releases
DefaultDirName={autopf}\Midstate
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; The release binaries need Windows 10 or later.
MinVersion=10.0
OutputBaseFilename=midstate-windows-amd64-setup
SetupIconFile=midstate-desktop.ico
UninstallDisplayIcon={app}\midstate-desktop.exe
UninstallDisplayName=Midstate Desktop
WizardStyle=modern
WizardImageFile=wizard-202x386.png,wizard-430x824.png
WizardSmallImageFile=wizard-small-58.png,wizard-small-97.png,wizard-small-159.png
Compression=lzma2/max
SolidCompression=yes
; A running copy is closed before its files are replaced, and is not
; reopened behind the user's back: the last page offers to open it.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#BinDir}\midstate-desktop.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#BinDir}\midstate.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Midstate Desktop"; Filename: "{app}\midstate-desktop.exe"
Name: "{autodesktop}\Midstate Desktop"; Filename: "{app}\midstate-desktop.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\midstate-desktop.exe"; Description: "{cm:LaunchProgram,Midstate Desktop}"; Flags: nowait postinstall skipifsilent
