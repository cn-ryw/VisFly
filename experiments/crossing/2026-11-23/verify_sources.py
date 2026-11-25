"""Verify the paper checkout, September scene file bytes and historical renderer."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

VISFLY = "7b0a4534b57051fd9b00ab4e561496f9abaacab1"
FIGFASHION = "8018a3e286238e8fa0932396051b9f7fe2bbc252"
DATASET = "134192bd343b618764e4497abb00e2dfd23a0b81"
RENDERER = "3f89600c942a4dee1866a589acd6e59741952a9f"
CATALOG_URL = f"https://huggingface.co/api/datasets/LiFanxing/VisFly-datasets/revision/{DATASET}?blobs=true"
SCENE_ASSETS = {
    "spy_datasets.scene_dataset_config.json": "Dataset registry",
    "configs/garage_crossing/garage_crossing_0.scene_instance.json": "Only selected scene instance",
    "configs/stages/garage_v1.stage_config.json": "Stage template",
    "stages/garage_v1.glb": "Shared render and collision geometry of the garage",
    "configs/lighting/garage_v1_0.lighting_config.json": "Lighting referenced by this scene",
    "configs/agents/debug_wall.object_config.json": "Rigid-object template used by both walls",
    "self_define_objects/wall.glb": "Geometry referenced by both walls",
    "configs/agents/DJI_Mavic_Mini_2.object_config.json": "Three drone object instances",
    "agents/DJI_Mavic_Mini_2.glb": "Render/collision asset named by the drone template",
}
SOURCE_PATHS = [
    "examples/crossing/rl.py", "envs/MultiNavigationEnv.py", "envs/multiDroneGymEnv.py",
    "envs/droneEnv.py", "envs/dynamics.py", "configs/example.json", "utils/maths.py",
    "utils/sceneDatasets.py", "utils/randomization.py", "utils/algorithms/ppo.py",
    "utils/policies/extractors.py", "utils/launcher.py", "environment.yml",
]


def write_json(path, value):
    part = path.with_suffix(path.suffix + ".partial")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    part.replace(path)


def git_data(kind, data):
    prefix = f"{kind} {len(data)}".encode() + b"\0"
    return hashlib.sha1(prefix + data).hexdigest()


def checksums(path, symlink=False):
    length = path.lstat().st_size if symlink else path.stat().st_size
    blob = hashlib.sha1(f"blob {length}".encode() + b"\0")
    sha = hashlib.sha256()
    if symlink:
        data = os.fsencode(os.readlink(path))
        blob.update(data)
        sha.update(data)
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                blob.update(block)
                sha.update(block)
    return length, blob.hexdigest(), sha.hexdigest()


def git(path, *arguments, raw=False):
    result = subprocess.check_output(["git", "-C", path, *arguments])
    return result if raw else result.decode().strip()


def checkout(path, expected):
    oid = git(path, "rev-parse", "HEAD")
    if oid != expected or git(path, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Unexpected checkout or modified tracked files in {path}")
    return dict(commit=oid, tree=git(path, "rev-parse", "HEAD^{tree}"), date=git(path, "show", "-s", "--format=%aI"),
                tracked_worktree_clean=True, path=str(path))


def check_assets(dataset, setup):
    catalog = json.loads((setup / "dataset-catalog.json").read_bytes())
    if catalog["sha"] != DATASET:
        raise ValueError("Unexpected Hugging Face catalogue revision")
    entries = {row["rfilename"][len("spy_datasets/"):]: row for row in catalog["siblings"]
               if row["rfilename"].startswith("spy_datasets/")}
    actual = {path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file()}
    if set(entries) != actual:
        raise ValueError(f"Catalogue and dataset differ: {sorted(set(entries) ^ actual)}")
    rows = []
    for name, remote in sorted(entries.items()):
        size, blob, sha = checksums(dataset / name)
        lfs = remote.get("lfs")
        if lfs:
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode()
            blob = git_data("blob", pointer)
            if sha != lfs["sha256"] or len(pointer) != lfs["pointerSize"] or size != lfs["size"]:
                raise ValueError(f"LFS target or pointer differs: {name}")
        if size != remote["size"] or blob != remote["blobId"]:
            raise ValueError(f"Content differs from the pinned revision: {name}")
        rows.append(dict(path=name, size=size, sha256=sha, git_blob=blob, lfs=bool(lfs)))
    write_json(setup / "dataset-files.json", rows)
    selected = [dict(**row, role=SCENE_ASSETS[row["path"]]) for row in rows if row["path"] in SCENE_ASSETS]
    if len(selected) != len(SCENE_ASSETS):
        raise ValueError("Missing selected crossing asset")
    return dict(repository=catalog["id"], revision=DATASET, date=catalog["lastModified"], catalog=CATALOG_URL,
                root=str(dataset), files=len(rows), bytes=sum(row["size"] for row in rows),
                git_blobs=sum(not row["lfs"] for row in rows), lfs_files=sum(row["lfs"] for row in rows),
                verified_all=True, listing="dataset-files.json", selected_scene_dependencies=selected,
                scope="All files below spy_datasets/. The repository-level .gitattributes is outside that directory.")


def observed_assets(run, assets):
    setup = run / "setup"
    log = setup / "eval-openat.log"
    smoke = run / "smoke/asset-probe"
    if not log.exists() or not (smoke / "complete.json").exists():
        return None
    root = Path(assets["root"])
    by_name = {r["path"]: r for r in json.loads((setup / assets["listing"]).read_text())}
    opened = defaultdict(int)
    raw = log.read_text()
    file_open = re.compile(r'openat\(AT_FDCWD, "([^"]+)", ([^)]*)\)\s*=\s*([0-9]+)')
    for path, flags, _ in file_open.findall(raw):
        if "spy_datasets/" not in path or "O_DIRECTORY" in flags:
            continue
        file_path = (run / "source" / path).resolve()
        if root in file_path.parents and file_path.is_file():
            opened[file_path.relative_to(root).as_posix()] += 1
    if set(SCENE_ASSETS) - set(opened):
        raise ValueError("Syscall trace does not contain every selected scene dependency")
    records = [dict(path=name, opened_count=opened[name], sha256=by_name[name]["sha256"],
                    role=SCENE_ASSETS.get(name, "Registry/config input")) for name in sorted(opened)]
    complete = json.loads((smoke / "complete.json").read_text())
    samples = list(smoke.glob("seed_*.json"))
    write_json(setup / "opened-assets.json", dict(
        proof="Successful openat calls for regular files inside the pinned dataset",
        raw_trace_sha256=checksums(log)[2], source_directory=str(run / "source"),
        smoke_manifest=complete, episodes=[json.loads(p.read_text()) for p in samples],
        per_file=records))
    return dict(listing="opened-assets.json", raw_trace=log.name, opened_distinct=len(opened),
                selected_scene_assets=list(SCENE_ASSETS), glb_opened=[r for r in opened if r.endswith(".glb")],
                navmesh_opened=[r for r in opened if r.endswith(".navmesh")],
                origin="Single original crossing environment, seed 997 with PPO_768; same scene/registry/renderer as training.")


def check_tree(tree, commit):
    if tree["truncated"] or tree["sha"] != RENDERER or git_data("commit", commit) != RENDERER:
        raise ValueError("Incomplete tree or unexpected renderer commit")
    children = defaultdict(list)
    for row in tree["tree"]:
        p = PurePosixPath(row["path"])
        children[str(p.parent)].append(row)
    expected = {row["path"]: row["sha"] for row in tree["tree"] if row["type"] == "tree"}
    expected["."] = commit.splitlines()[0].decode().removeprefix("tree ")
    for path, rows in children.items():
        def key(row):
            return PurePosixPath(row["path"]).name.encode() + (b"/" if row["type"] == "tree" else b"")
        content = bytearray()
        for row in sorted(rows, key=key):
            mode = row["mode"].lstrip("0")
            content.extend(f"{mode} {PurePosixPath(row['path']).name}".encode() + b"\0" + bytes.fromhex(row["sha"]))
        if git_data("tree", content) != expected[path]:
            raise ValueError(f"Renderer tree does not match upstream Git object: {path}")
    return expected["."]


def check_renderer_blob(root, row, reference=None):
    entry = root / row["path"]
    link = row["mode"] == "120000"
    if entry.is_symlink() != link:
        raise ValueError(f"Renderer source type differs: {entry}")
    converted = checksums(entry, symlink=link)[1] != row["sha"]
    if converted:
        if reference is None or link:
            raise ValueError(f"Renderer source bytes differ: {entry}")
        attrs = git(reference, "check-attr", "eol", "--", row["path"]).rsplit(": ", 1)[-1]
        actual = entry.read_bytes()
        if (attrs != "crlf" or actual != (reference / row["path"]).read_bytes()
                or git_data("blob", actual.replace(b"\r\n", b"\n")) != row["sha"]):
            raise ValueError(f"Renderer source differs after applying checkout attributes: {entry}")
    if not link and bool(entry.stat().st_mode & 0o111) != (row["mode"] == "100755"):
        raise ValueError(f"Renderer source mode differs: {entry}")
    return converted


def check_submodule(reference, copy, revision, listed, prefix, checkout_eol):
    if git(reference, "rev-parse", "HEAD") != revision:
        raise ValueError(f"Incorrect reference submodule HEAD: {reference}")
    tree = git(reference, "ls-tree", "-rz", revision, raw=True)
    count = 0
    for line in tree.split(b"\0"):
        if not line:
            continue
        desc, rawpath = line.split(b"\t", 1)
        mode, kind, sha = desc.decode().split()
        name = os.fsdecode(rawpath)
        if kind == "commit":
            count += check_submodule(reference / name, copy / name, sha, listed, prefix / name, checkout_eol)
        else:
            if check_renderer_blob(copy, dict(path=name, sha=sha, mode=mode), reference):
                checkout_eol.append((prefix / name).as_posix())
            count += 1
    listed.append(dict(path=prefix.as_posix(), revision=revision, tracked_files_including_nested=count))
    return count


def check_renderer(run, original):
    setup = run / "setup"
    library = (setup / "renderer-lib").resolve()
    build = library.parents[1]
    commit = (setup / "renderer-commit.raw").read_bytes()
    tree = json.loads((setup / "renderer-tree.json").read_bytes())
    tree_oid = check_tree(tree, commit)
    areas = {"src", "src_python", "data", "tests", "tools", "conda-build"}
    files = [r for r in tree["tree"] if r["type"] in {"blob", "commit"} and
             (r["path"].split("/")[0] in areas or "/" not in r["path"])]
    submods = []
    n_subfiles = 0
    checkout_eol = []
    for row in files:
        if row["type"] == "commit":
            relative = Path(row["path"])
            n_subfiles += check_submodule(original / relative, build / relative, row["sha"], submods, relative, checkout_eol)
        else:
            check_renderer_blob(build, row)
    binary, = (library / "habitat_sim/_ext").glob("habitat_sim_bindings*.so")
    log = build.parent / "renderer_2024_build.log"
    options = ["BUILD_GUI_VIEWERS", "BUILD_WITH_BULLET", "BUILD_WITH_CUDA", "BUILD_WITH_BACKGROUND_RENDERER",
               "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_MAKE_PROGRAM"]
    cache = (build / "build/CMakeCache.txt").read_text()
    settings = {key: re.search(rf"^{key}:\w+=(.+)$", cache, re.M).group(1) for key in options}
    pyfiles = list((library / "habitat_sim").rglob("*.py"))
    for path in pyfiles:
        if path.read_bytes() != (build / "src_python" / path.relative_to(library)).read_bytes():
            raise ValueError(f"Installed Python wrapper differs: {path}")
    return dict(repository="https://github.com/Fanxing-LI/habitat-sim", revision=RENDERER,
                tree=tree_oid, git_commit_file="renderer-commit.raw", git_tree_manifest="renderer-tree.json",
                original_working_trees=str(original), build_source=str(build), build_inputs_and_top_gitlinks=len(files),
                verified_submodule_files=n_subfiles, submodules=submods, git_eol_crlf_checkouts=checkout_eol,
                copied_python_wrappers=len(pyfiles),
                binding=str(binary), binding_sha256=checksums(binary)[2], build_settings=settings, build_log=str(log),
                build_log_sha256=checksums(log)[2], cmake_cache_sha256=checksums(build / "build/CMakeCache.txt")[2])


def environment(executable, library):
    snippet = """
import importlib.metadata as md
import json, platform, sys
import habitat_sim
import habitat_sim._ext.habitat_sim_bindings as bindings
import torch
packages=['numpy','torch','torchvision','gym','gymnasium','stable-baselines3','tensorboard','numpy-quaternion','scipy','magnum']
print(json.dumps(dict(python=sys.version, executable=sys.executable, kernel=platform.release(),
  cuda_torch=torch.version.cuda, cudnn_torch=torch.backends.cudnn.version(),
  modules={p:md.version(p) for p in packages}, binding=bindings.__file__, habitat_module=habitat_sim.__file__,
  habitat_version_declared=getattr(habitat_sim,'__version__',None))))
"""
    variables = dict(os.environ, PYTHONPATH=str(library), PYTHONDONTWRITEBYTECODE="1")
    value = json.loads(subprocess.check_output([executable, "-c", snippet], env=variables, text=True))
    if library.resolve() not in Path(value["binding"]).resolve().parents:
        raise ValueError("Runtime imported a different renderer")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                   "--format=csv,noheader,nounits"], text=True).strip().rsplit(", ", 2)
    value["gpu"] = dict(name=gpu[0], driver=gpu[1], total_memory_mib=int(gpu[2]))
    value["os"] = Path("/etc/os-release").read_text().splitlines()[0]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--submodule-references", required=True, type=Path,
                        help="Habitat-Sim checkout containing the recorded dependency gitlinks and nested gitlinks")
    parser.add_argument("--visfly-python", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    setup = run / "setup"
    source = run / "source"
    visfly = checkout(source, VISFLY)
    visfly["upstream"] = "https://github.com/SJTU-ViSYS-team/VisFly"
    visfly["selected_sources_sha256"] = {p: checksums(source / p)[2] for p in SOURCE_PATHS}
    fig = checkout(source / "utils/FigFashion", FIGFASHION)
    if git(source, "rev-parse", "HEAD:utils/FigFashion") != FIGFASHION:
        raise ValueError("Unexpected FigFashion gitlink")
    assets = check_assets((source / "datasets/spy_datasets").resolve(), setup)
    assets["observed_opened_assets"] = observed_assets(run, assets)
    renderer = check_renderer(run, args.submodule_references.resolve())
    runtime = environment(args.visfly_python.resolve(), setup / "renderer-lib")
    record = dict(visfly=visfly, figfashion=fig, dataset=assets, renderer=renderer, runtime=runtime,
                  paper_claim="These revisions reproduce the public paper-era artifacts. The paper does not identify the commits or the raw run/checkpoint that generated its Figure 7.")
    write_json(setup / "source-manifest.json", record)
    print(f"Verified {assets['files']} assets, {renderer['build_inputs_and_top_gitlinks']} renderer inputs/gitlinks,"
          f" {renderer['verified_submodule_files']} dependency files and the loaded renderer {renderer['binding_sha256']}")


if __name__ == "__main__":
    main()
