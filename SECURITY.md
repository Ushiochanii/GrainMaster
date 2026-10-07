# Security and local credentials

GrainMaster is designed as a local workstation.

- Do not commit API keys, tokens, private image collections, or runtime state.
- Paper-label API keys are configured by the user and are not bundled with the release.
- Keep artifacts, local uploads, and credential files out of version control.
- The default web server binds to 127.0.0.1.

If you discover a credential or private dataset in a public commit, revoke the credential
first, then remove the material from Git history before publishing again.
