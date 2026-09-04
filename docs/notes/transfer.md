# Moving a corpus (and the code) to the training rig

Development happens on a laptop / the work PC; prod training happens on the 5090 rig.
`signalml ship` is how a selected corpus and the exact commit that produced it get from
one to the other over the LAN. `signalml doctor` is how the receiving machine proves it
can actually run what arrived.

## Why a tool and not `robocopy`

Throughput is not the problem. Over 1 GbE (~110 MB/s real) a 35 GB haul is about six
minutes; `robocopy /MIR /MT:16 /Z` moves those bytes fine. Three things it cannot do,
and all three are why this exists:

1. **Know what to send.** Of ~74 GB in a populated `DATA_ROOT`, the rig needs roughly
   half: `clean/vocals.wav` and `align/`, not the non-vocal Demucs stems, not the
   pre-clean `stems/vocals.wav`, not the re-downloadable source corpora. Selection is a
   manifest question, and the manifest is what `dataset build` already answers it with.
2. **Prove the bytes arrived.** Every item carries a sha256. `ship verify` is the gate
   you run before committing a GPU to a multi-day run — a truncated wav should cost
   seconds, not eight hours.
3. **Carry the code.** This repo has **no git remote**. The rig cannot clone from
   anywhere, so the shipment carries a git bundle, and that bundle is the rig's clone
   *and* its update path forever after.

`robocopy` remains the right tool for a one-off dump of a directory you already
understand. It is not the right tool for "give the rig what recipe X trains on".

## Model

A shipment is a **plan** plus a **transport**.

The plan (`DATA_ROOT/ship/<name>/SHIP.json`) lists every file with its destination-
relative path, size and sha256, plus git provenance and the song ids selected. Nothing
is copied at plan time — only generated files (the manifest subset, the bundle) are
staged. Re-planning is cheap: a `(size, mtime_ns) -> sha256` cache under
`DATA_ROOT/ship/hashcache.json` means a second plan over 35 GB does no rehashing.

The transport is a stdlib HTTP server. One port, one firewall rule, no dependency, and
it works unchanged if the receiver later becomes a Linux box or a cloud instance.

**The rig pulls.** It is the machine you are sitting at when you start a run, and
resume/verify logic belongs where the bytes land. The same asymmetry runs in reverse
later: the rig serves, the laptop pulls checkpoints back.

### Selection modes

| `--what` | Contents | Use when |
|---|---|---|
| `dataset` | `datasets/<name>/` — wavs, `transcriptions.csv`, dictionary, trainer config, card | the recipe is settled and you only want to train |
| `rebuildable` | `clean/vocals.wav`, `align/`, `analysis.json`, `score.json` for the songs the recipe selects | first haul — the rig can then rebuild datasets under new recipes without another transfer |
| `full` | the above plus `stems/`, and `raw/` with `--with-raw` | you intend to re-separate or re-align on the rig |

`--with-features` adds `features/vocals.npz` (~5 MB/song). Skip it: features are
profile-bound and the rig recomputes them faster than it can receive them.

Selection for `rebuildable`/`full` runs the *same* `_select` that `dataset build` uses,
so shipping and training can never disagree about what "the corpus" means.

### Manifest and provenance

The manifest travels as a **subset** of the records whose audio is in the shipment — the
receiver must not learn about songs whose files did not come with them, or its
`status.*` flags would be lies. On arrival it is **upserted by id**, so successive
shipments accumulate rather than clobber.

Planning with `--with-code` refuses a dirty worktree. Checkpoints record a git hash;
the rig must never run code that does not correspond to a commit. `--allow-dirty`
overrides it and captures `git diff HEAD` as `.ship/dirty.patch`, which is copied to the
rig but never applied automatically.

### Integrity and resume

Downloads land as `<file>.part`, resume with an HTTP `Range` request, are hashed as they
stream, and only then get `os.replace`d into position. An item whose destination already
matches size + sha256 is skipped (a receipts file avoids rehashing what was verified on
a previous run). Re-running an interrupted pull is always the correct move.

### Security posture

Home LAN, sized to match. The server is read-only, ephemeral, serves *only* files the
plan lists, and addresses them by an opaque key derived from the path — traversal is not
expressible in the protocol. A random token gates every route, and a bad token is
indistinguishable from a bad path. This is not TLS and it is not for the open internet.

## Recipes

Sender (the machine holding `DATA_ROOT`):

```powershell
signalml ship plan --data-root Y:\DATA_ROOT --what rebuildable
signalml ship serve --data-root Y:\DATA_ROOT --name full_acoustic_v1
```

`serve` prints the exact `ship pull` line, including the LAN address and token.

Receiver (the rig), first time — the repo does not exist yet:

```powershell
signalml ship pull http://10.0.0.144:8770/<token> --data-root D:\DATA_ROOT --repo-dir D:\SignalML
cd D:\SignalML
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_rig.ps1 -DataRoot D:\DATA_ROOT
```

Every time after, the same `pull` command fetches only what changed and fast-forwards
the repo. Before a run:

```powershell
signalml ship verify --data-root D:\DATA_ROOT --name full_acoustic_v1
signalml doctor --data-root D:\DATA_ROOT --plan D:\DATA_ROOT\ship\full_acoustic_v1\SHIP.json
```

`doctor` exits non-zero if anything FAILs, so it can gate a script. Warnings are
informational (MFA, for instance, is only needed if you align on the rig).

## Gotchas

- **VPNs.** ProtonVPN / NordLynx with "block LAN traffic" enabled kill both this and
  SMB. Disconnect, or exclude the local subnet.
- **Virtual adapters.** A WSL `vEthernet` adapter can win the default-route guess that
  picks the advertised address. Pass `--host <LAN IP>` explicitly if the printed URL
  looks wrong.
- **Firewall.** The serving machine needs one inbound rule; `serve` prints the
  `New-NetFirewallRule` command. Private profile only.
- **`uv sync` reverts CUDA wheels.** It resolves CPU torch on Windows. `doctor` fails
  loudly on this (it checks `torch.version.cuda` *and* that the built arch list covers
  the installed GPU); `bootstrap_rig.ps1` re-applies the swap and is safe to re-run.
