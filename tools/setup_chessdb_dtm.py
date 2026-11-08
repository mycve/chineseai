"""Build the local ChessDB probe and fetch the file index. Run on Linux or WSL."""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


SOURCE = "https://github.com/noobpwnftw/chessdb.git"
REVISION = "ab6c33133dba40e46a6f9828696c959fb4940221"
MIRROR = "rsync://ftp.chessdb.cn/ftp/pub/EGTB_DTM/"
DEST = Path(__file__).resolve().parents[1] / "data" / "chessdb-dtm"


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def build(source):
    probe_dir = source / "extensions" / "ccegtbprobe"
    upstream = (probe_dir / "ccegtbprobe.cpp").read_text(encoding="utf-8")
    # Keep the upstream index and file reader, replacing only the PHP entry point.
    reader = upstream[upstream.index("#include <unistd.h>"):]
    start = reader.index('#include "ccegtbprobe_arginfo.h"')
    end = reader.index("static inline int uncompress_lzma", start)
    reader = reader[:start] + reader[end:]
    start = reader.index("PHP_MINIT_FUNCTION(ccegtbprobe)")
    end = reader.index("static bool read_egtb_value_cached", start)
    reader = reader[:start] + reader[end:]
    reader = reader.replace('"/data/EGTB_DTM/"', '"./"')
    reader = reader.replace(
        "if (!slot || slot->error || !slot->done) return false;",
        "if (!slot || slot->error || !slot->done || "
        "(blk_offset << 1) + sizeof(U16) > slot->decoded_size) return false;",
    )
    reader += r"""

int main(int argc, char** argv) {
    if (argc != 2) return 1;
    init();
    piece_index_init();
    if (!group_init()) return 1;

    Position position;
    if (!position.from_fen(argv[1]) || !position.is_legal() ||
        position.number[WhitePawn] >= 4 || position.number[BlackPawn] >= 4) {
        return 1;
    }
    ProbeSession session;
    U16 raw;
    U64 flags;
    if (!probe_egtb(position, true, raw, flags, &session)) return 1;
    std::printf("%d\n", decode_root_dtm(raw));
}
"""
    (probe_dir / "probe.cpp").write_text(reader, encoding="utf-8")
    piece_set = probe_dir / "piece_set.cpp"
    piece_set.write_text(
        piece_set.read_text(encoding="utf-8").replace("/home/apache/GROUP.dat", "GROUP.dat"),
        encoding="utf-8",
    )
    objects = []
    for name in ("Alloc", "CpuArch", "LzmaDec"):
        obj = f"{name}.o"
        run("cc", "-O2", "-c", f"LZMA/{name}.c", "-o", obj, cwd=probe_dir)
        objects.append(obj)
    run(
        "c++", "-std=c++17", "-O2", "-pthread", "-mpopcnt",
        "-ffunction-sections", "-fdata-sections", "-Wl,--gc-sections",
        "probe.cpp", "piece_set.cpp", "../ccboard/xiangqi.cpp",
        "LZMA/LzmaLib.c", *objects, "-o", "probe", cwd=probe_dir,
    )
    for name in ("GROUP.dat", "probe"):
        temporary = DEST / f"{name}.new"
        shutil.copy2(probe_dir / name, temporary)
        temporary.replace(DEST / name)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chessdb-probe-") as directory:
        source = Path(directory)
        run("git", "init", "-q", cwd=source)
        upstream = str(Path(sys.argv[1]).resolve()) if len(sys.argv) == 2 else SOURCE
        run("git", "fetch", "-q", "--depth=1", upstream, REVISION, cwd=source)
        run("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=source)
        build(source)

    listing = subprocess.check_output(
        ["rsync", "--list-only", "--contimeout=15", "--timeout=30", MIRROR],
        text=True,
    )
    files = sorted(set(re.findall(r"\bK[ABKRNCP]+\.lzdtm$", listing, re.MULTILINE)))
    if not files:
        raise RuntimeError("ChessDB DTM mirror returned an empty index")
    index = DEST / "files.txt.new"
    index.write_text("\n".join(files) + "\n", encoding="ascii")
    index.replace(DEST / "files.txt")
    print(f"ChessDB DTM probe ready ({len(files)} indexed tables)")


if __name__ == "__main__":
    main()
