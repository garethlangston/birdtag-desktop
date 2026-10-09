# Privacy in this fork (uk-models branch)

This personal fork keeps all analysis on your computer, as upstream Kestrel does,
and also turns off every *automatic* network call. **On a default install, starting
Kestrel or analysing folders sends no network requests.** Anything listed below
happens only when you switch it on in Settings or click the matching button.

## What changed from upstream

| Behaviour | Upstream default | This fork |
|---|---|---|
| Crash reports (`crash_reports_enabled`) | on (opt-out) | **off** (opt-in in Settings) |
| Daily "app open" ping (`send_app_open_telemetry` in `visualizer.py`) | sent once terms accepted | only if *Share anonymous usage analytics* is ticked |
| One-off installation ping (`api_bridge.agree_to_legal`) | sent on accepting terms | only if analytics is ticked |
| Analysis-completion ping (`queue_manager._send_folder_analytics`) | always sent | only if analytics is ticked (local "photos analysed" counters still update) |
| Per-folder analytics | opt-in | unchanged (opt-in) |
| Startup version check (`projectkestrel.org/version_v2.json`, `version.json`) | every start | only if **Settings → Check for new versions at startup** is ticked |
| Startup terms check (`projectkestrel.org/legal.json`) | every start | only with that same setting; otherwise the existing offline fallback is used |
| Sign-in (OAuth / Clerk), Perch, Cloud Compute | user-initiated | unchanged: user-initiated only, never prompted at startup |

Upstream's other pings are *not* restricted to packaged builds: running from source
(`python analyzer/visualizer.py`) would send them too, which is why the opt-in gates
above live in the calling code. Crash reports are additionally limited to packaged
builds by `kestrel_telemetry.is_frozen()`.

## Network endpoints still in the code

All of these are HTTPS and are only contacted after a user action or opt-in.

| Host | Code | When it is contacted |
|---|---|---|
| `api.projectkestrel.org` | `kestrel_telemetry.py`, `api_bridge.py` | Analytics or crash reports **if opted in**. *Send feedback* and the post-crash *recovery report* dialog only when you click send. |
| `projectkestrel.org` (`version_v2.json`, `version.json`, `legal.json`) | `api_bridge.py`, `js/auth.js` | Startup version and terms checks, **only if "Check for new versions at startup" is on** |
| `projectkestrel.org` (help, privacy, terms, support pages) | `visualizer.html`, `js/*.js` | Links. Opened in your browser only when clicked |
| `clerk.projectkestrel.org`, `myaccount.projectkestrel.org` | `oauth_client.py`, `js/auth.js` | Sign-in / account pages, only when you click *Sign in*. Signed-in users also get a profile lookup at startup. |
| `auth.projectkestrel.org` | `auth_client.py`, `api_bridge.py` | Token exchange or refresh for a signed-in account |
| `perchapi.projectkestrel.org`, `perch.projectkestrel.org` | `perch_uploader.py`, `api_bridge.py`, `js/perch.js` | Publishing to Perch (paid add-on), only when you publish |
| `cloudcompute.projectkestrel.org` | `cloud_compute_client.py` | Cloud Compute jobs (paid add-on), only when you submit one. Polling runs only while you have jobs. |
| `appleid.apple.com` | `mac_apple_signin.py` | macOS *Sign in with Apple*, when clicked (not used on Windows) |
| `github.com` | `visualizer.html`, `_speciesnet_ensemble.py` | Links and licence comments only. No requests |

Model downloads: the UK/Europe/World bird model (EVA-02) is **not** downloaded by
Kestrel. You export or copy it yourself (see `FORK-NOTES.md`), so there is no
download endpoint for it in the code.

## Checking for yourself

```
grep -rnE "urlopen|requests\.|http.client" analyzer --include=*.py | grep -v tests/
grep -rn "https://" analyzer/js analyzer/*.html
```
