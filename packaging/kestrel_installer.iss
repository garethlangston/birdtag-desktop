; Inno Setup script for the fork (unsigned, per-user install).
; Every product-specific value is passed in from analyzer/branding.json by CI
; (see .github/workflows/fork-windows-build.yml), e.g.
;   ISCC /DMyAppName="Birdtag Desktop" /DExeName=BirdtagDesktop /DAppGuid=... kestrel_installer.iss
; Upstream Project Kestrel names, ids, URLs and artwork are not used.

#ifndef MyAppName
  #define MyAppName "Birdtag Desktop"
#endif
#ifndef ExeName
  #define ExeName "BirdtagDesktop"
#endif
#ifndef AppGuid
  #define AppGuid "00000000-0000-0000-0000-000000000000"
#endif
#ifndef MyAppPublisher
  #define MyAppPublisher "Birdtag Desktop contributors"
#endif
#ifndef MyAppURL
  #define MyAppURL "https://github.com/garethlangston/birdtag-desktop"
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0-dev"
#endif

[Setup]
AppId={{{#AppGuid}}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppVerName={#MyAppName} {#AppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\dist\installer
OutputBaseFilename={#ExeName}-{#AppVersion}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
DisableProgramGroupPage=yes
SetupIconFile=..\assets\logo.ico
UninstallDisplayIcon={app}\{#ExeName}\{#ExeName}.exe
WizardImageFile=..\assets\installer-side.png
WizardSmallImageFile=..\assets\installer-small.png
LicenseFile=..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Desktop shortcuts:"; Flags: checkedonce

[Files]
Source: "..\analyzer\dist\{#ExeName}\*"; DestDir: "{app}\{#ExeName}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\FORK-NOTES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\PRIVACY-FORK.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#ExeName}\{#ExeName}.exe"; WorkingDir: "{app}\{#ExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#ExeName}\{#ExeName}.exe"; WorkingDir: "{app}\{#ExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}\{#ExeName}.exe"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent unchecked
