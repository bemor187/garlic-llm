"""Copy model files to the PS4 over FTP (GoldHEN's FTP server; set PS4_IP, see build.py).

  python upload_model.py models/a.bin [models/b.bin ...]    -> /data/<file>: the built-in model's files
  python upload_model.py --model models/<name> [--name=<name>] [--dry-run]
      -> /data/garlic/models/<name>/: a folder pc/convert_9b_q4s.py --out made (another Qwen3.5-9B), which the app
         then offers at start. <name> is the folder's name unless --name gives another: 1 to 32 bytes, without '/'
         or '\\', not starting with '.'. The folder is checked first, as the app checks it (chat/models.c): its
         weights file against chat/assets_9b/model.idx, and its model.idx and MTP head if it has them. --dry-run
         only checks it and says what would be copied.

Each file goes up as NAME.part and is renamed when complete; LIST checks the size, since GoldHEN's SIZE truncates to
32 bits."""
import ftplib
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build as b  # noqa: E402

MODELS_DIR = "/data/garlic/models"   # chat/models.h
ASSETS = b.ROOT / "chat" / "assets_9b"


def upload_plain(paths):
    for p in map(Path, paths):
        t0 = time.time()
        ftp = b.ps4_ftp(timeout=60)
        with open(p, "rb") as f:
            ftp.storbinary(f"STOR /data/{p.name}.part", f, blocksize=4 << 20)
        lines = []
        ftp.retrlines(f"LIST /data/{p.name}.part", lines.append)
        size = int(lines[0].split()[4]) if lines else -1
        if size != p.stat().st_size:
            sys.exit(f"{p.name}: uploaded {size} bytes of {p.stat().st_size}")
        try:
            ftp.delete(f"/data/{p.name}")   # (the older copy of the same file, replaced by the new one)
        except ftplib.error_perm:
            pass
        ftp.rename(f"/data/{p.name}.part", f"/data/{p.name}")
        ftp.quit()
        print(f"{p.name}: {size / 2**20:.0f} MiB in {time.time() - t0:.0f} s ({size / 2**20 / (time.time() - t0):.1f} "
              "MiB/s)", flush=True)


# ---- a model folder ---------------------------------------------------------------------------------------------

def name_ok(name):
    """As chat/models.c model_name_ok: 1 to 32 bytes of UTF-8, no control characters, '/' or '\\', no leading '.'."""
    try:
        raw = name.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return (0 < len(raw) <= 32 and not name.startswith(".") and "/" not in name and "\\" not in name and
            not any(ord(c) < 0x20 or ord(c) == 0x7F for c in name))


def idx_file(idx, at):   # the file name at the end of an index's weights path (128 bytes at `at`)
    return idx[at:at + 128].split(b"\0")[0].decode().rsplit("/", 1)[-1]


def check_folder(folder):
    """The folder's files as the app checks them: [(file, what it is)] to copy, or the program ends saying what does
    not match."""
    builtin = (ASSETS / "model.idx").read_bytes()
    weights = idx_file(builtin, 52)
    n, flags = struct.unpack_from("<I", builtin, 188)[0], struct.unpack_from("<I", builtin, 220)[0]
    recs = [struct.unpack_from("<QQIIQI", builtin, 224 + 36 * i) for i in range(n)]
    end = max(off + size for off, size, *_ in recs)
    w = folder / weights
    if not w.is_file():
        sys.exit(f"{folder}: no {weights} (pc/convert_9b_q4s.py --out makes the folder)")
    if w.stat().st_size != end:
        sys.exit(f"{w}: {w.stat().st_size} bytes, the built-in model's file has {end}: incomplete, or another "
                 "conversion")
    files = [(w, "the weights")]
    own = folder / "model.idx"
    if own.is_file():
        mine = own.read_bytes()
        if len(mine) != len(builtin) or mine[:52] != builtin[:52] or mine[180:] != builtin[180:]:
            sys.exit(f"{own} differs from chat/assets_9b/model.idx (beyond the weights path): another shape, format "
                     "or conversion, which the app does not load")
        files.append((own, "its index"))
    if flags & 4:   # the vocabulary table: every entry a token (below the input embedding's rows)
        off, size, rows = recs[-1][0], recs[-1][1], recs[0][2]
        with open(w, "rb") as f:
            f.seek(off)
            ids = struct.unpack(f"<{size // 4}I", f.read(size))
        if max(ids) >= rows:
            sys.exit(f"{w}: its vocabulary table has an entry that is not a token ({max(ids)})")
    mtp_idx = ASSETS / "mtp.idx"
    if mtp_idx.is_file():
        bm = mtp_idx.read_bytes()
        m = folder / idx_file(bm, 8)
        if m.is_file():
            total = sum(struct.unpack_from("<QQ", bm, 136))
            if m.stat().st_size != total:
                sys.exit(f"{m}: {m.stat().st_size} bytes, the MTP head has {total}")
            files.append((m, "its MTP head"))
            own = folder / "mtp.idx"
            if own.is_file():
                mine = own.read_bytes()
                if len(mine) != len(bm) or mine[:8] != bm[:8] or mine[136:] != bm[136:]:
                    sys.exit(f"{own} differs from chat/assets_9b/mtp.idx (beyond the weights path)")
                files.append((own, "its MTP head's index"))
        else:
            print(f"{folder.name}: no MTP head of its own; the app drafts with the built-in one")
    return files


def listed_size(ftp, name):   # the size of a file in the current folder, from LIST (SIZE stops at 32 bits)
    lines = []
    ftp.retrlines("LIST", lines.append)
    for line in lines:
        parts = line.split(None, 8)
        if len(parts) == 9 and parts[8] == name:
            return int(parts[4])
    return -1


def upload_model(folder, name, dry):
    if not folder.is_dir():
        sys.exit(f"{folder}: not a folder")
    if not name_ok(name):
        sys.exit(f"{name!r}: the app lists folders whose names have 1 to 32 bytes, without '/' or '\\', not starting "
                 "with '.': give another with --name=")
    files = check_folder(folder)
    remote = f"{MODELS_DIR}/{name}"
    for p, what in files:
        print(f"{p} ({what}, {p.stat().st_size / 2**20:.0f} MiB) -> {remote}/{p.name}")
    if dry:
        return
    for p, _ in files:
        t0 = time.time()
        ftp = b.ps4_ftp(timeout=60)
        for d in ("/data/garlic", MODELS_DIR, remote):
            try:
                ftp.mkd(d)
            except ftplib.error_perm:   # (already there)
                pass
        ftp.cwd(remote)
        with open(p, "rb") as f:
            ftp.storbinary(f"STOR {p.name}.part", f, blocksize=4 << 20)
        size = listed_size(ftp, p.name + ".part")
        if size != p.stat().st_size:
            sys.exit(f"{p.name}: uploaded {size} bytes of {p.stat().st_size}")
        try:
            ftp.delete(p.name)   # (an older copy, replaced by the new one)
        except ftplib.error_perm:
            pass
        ftp.rename(p.name + ".part", p.name)
        ftp.quit()
        s = max(time.time() - t0, 1e-3)
        print(f"{p.name}: {size / 2**20:.0f} MiB in {s:.0f} s ({size / 2**20 / s:.1f} MiB/s)", flush=True)
    print(f"done: the app offers {name} when it starts")


def main(argv):
    opts = {a.split("=", 1)[0]: a.split("=", 1)[1] if "=" in a else "" for a in argv if a.startswith("--")}
    rest = [a for a in argv if not a.startswith("--")]
    if "--model" in opts:
        folder = Path(opts["--model"] or (rest.pop(0) if rest else "."))
        upload_model(folder, opts.get("--name") or folder.resolve().name, "--dry-run" in opts)
    elif rest:
        upload_plain(rest)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
