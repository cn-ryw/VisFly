"""Export one new Excel sheet without reserializing existing worksheets or formula caches."""
import argparse
import hashlib
import os
from pathlib import Path
import posixpath
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree as et
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from present_results import ARM, ARMS, OUTCOME, SPLIT, load_experiment, paired_counts, read_json, role


SHEET = "三机机间避碰对照"
TEMPLATE = "论文同期三机基线"
MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE = "http://schemas.openxmlformats.org/package/2006/relationships"
TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
WB, REL, CT = "xl/workbook.xml", "xl/_rels/workbook.xml.rels", "[Content_Types].xml"
EXECUTION = {"success": "执行通过", "collision": "执行碰撞", "timeout": "执行超时"}
WIDTHS = [19, 16, 15, 23, 12, 13, 13, 13, 13, 16, 17, 18, 18, 20, 17, 20, 17]


def add_panel(book, run, spec, best, cohorts):
    template = book[TEMPLATE]
    sheet = book.create_sheet(SHEET)
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "D5"
    for index, width in enumerate(WIDTHS, 1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    line = 0

    def row(values, style="A2", height=21, merge=False):
        nonlocal line
        line += 1
        if merge:
            sheet.merge_cells(start_row=line, start_column=1, end_row=line, end_column=len(WIDTHS))
        for col, value in enumerate(values, 1):
            cell = sheet.cell(line, col, value)
            cell._style = template[style]._style
        sheet.row_dimensions[line].height = height
        return line

    final = spec["checkpoints"][-1]
    ref = spec["evaluation"]["reference"]
    row(["三机机间避碰对照  |  安全起点与 20 cm 终止"], "A1", 36, True)
    row([f"训练种子 {spec['train_seed']}。两组均以机间中心距 ≥ 20 cm 的队形起飞。机间终止组在任意两机中心距 < 20 cm 时结束团队训练回合。"], merge=True)
    row(["全部评估使用相同起点、确定性推理和共同终止规则。严格通过要求三机均经墙缝越过 x = 10 m，且全程无几何接触。首次接触可同时涉及不同对象。"], merge=True)
    row(["评估集 / 种子", "训练组", "智能体交互", "权重用途", "团队回合", "严格通过", "严格碰撞", "超时", "绕行 / 未完成",
         "首次机间接触", "飞行中机间终止", "通过率 95% 区间", "平均控制步", "最小起点间距（cm）", "最近距中位数（cm）",
         "最近距 P10（cm）", "权重训练耗时（s）"], "A4", 35)
    checks = {}
    for (split, arm, step), cohort in cohorts.items():
        s = cohort.summary
        cnt = s["counts"]
        if sum(cnt.values()) != s["episodes"]:
            raise ValueError(f"Incomplete strict-outcome counts in {cohort.folder}")
        low, high = s["strict_success_wilson95"]
        seconds = read_json(Path(s["checkpoint"]).with_suffix(".json"))["elapsed_s"]
        checks[row([f"{SPLIT[split]} {s['first']}–{s['last']}", ARM[arm], step, role(arm, step, final, best),
                    s["episodes"], s["strict_successes"], cnt.get("collision", 0), cnt.get("timeout", 0),
                    f"{cnt.get('bypass', 0)} / {cnt.get('incomplete', 0)}", s["peer_first"], s["any_peer_stops"],
                    f"{low:.1%}–{high:.1%}", round(s["mean_control_steps"], 2), round(100*s["minimum_start_m"], 3),
                    round(100*s["min_peer_quantiles_m"]["0.5"], 3), round(100*s["min_peer_quantiles_m"]["0.1"], 3),
                    round(seconds, 3)])] = s["strict_successes"]
    n_old = ref["old_heldout_seeds"][1] - ref["old_heldout_seeds"][0] + 1
    row([f"11 月 23–24 日历史留出集为 {ref['old_heldout_strict']}/{n_old} 严格通过，使用旧起点和种子 {ref['old_heldout_seeds'][0]}–{ref['old_heldout_seeds'][1]}。上表的历史末权重在本轮统一起点重新评估。"], merge=True)
    row(["训练期间的起点验收与机间终止"], "A1", 30, True)
    row(["训练组", "智能体交互", "已接受队形数", "已丢弃候选队形数", "最小起点间距（cm）", "机间终止次数"], "A4", 35)
    for arm in ARMS:
        stats = read_json(run / "train" / arm / "safety-complete.json")
        if stats["min_accepted_m"] < .20 or stats["terminate"] != (arm == "peer"):
            raise ValueError(f"Invalid training reset or termination evidence for {arm}")
        row([ARM[arm], final, stats["accepted_teams"], stats["rejected_teams"], round(stats["min_accepted_m"]*100, 8),
             stats["peer_terminals"] if stats["terminate"] else "未启用"])
    row(["训练起点汇总和逐次累计记录见 train/{control,peer}/safety-complete.json、resets.csv。候选队形若有任意两机的中心距低于 20 cm，整队重新抽样。"], merge=True)
    row(["同种子配对  |  各格计数相加等于该评估集的团队数"], "A1", 30, True)
    row(["评估集", "对照组交互", "机间终止组交互", "比较用途", "两组均通过", "仅对照通过", "仅机间终止组通过", "两组均未通过"], "A4", 35)
    for split in SPLIT:
        available = [(s, s) for s in spec["checkpoints"] if (split, "peer", s) in cohorts]
        if split == "holdout":
            available = [(final, final)]
            chosen = tuple(best[a] for a in ARMS)
            if chosen != (final, final):
                available.append(chosen)
        for a, b in available:
            is_chosen = (a, b) == tuple(best[arm] for arm in ARMS)
            name = "末权重兼开发最优" if a == b == final and is_chosen else (
                   "末权重" if a == b == final else ("开发最优" if is_chosen else "同预算"))
            row([SPLIT[split], a, b, name, *paired_counts(cohorts[split, "control", a], cohorts[split, "peer", b])])
    row(["证据目录与权重  |  相对路径均相对于下行的实验目录"], "A1", 30, True)
    row([str(run)], merge=True)
    row(["训练曲线位于 train/control/training.csv、train/peer/training.csv。选模使用 selection.json，协议与验收口径见 setup/protocol.json。"], merge=True)
    for (split, arm, step), cohort in cohorts.items():
        s = cohort.summary
        row([f"{SPLIT[split]} · {ARM[arm]} · {step:,} 条交互    {cohort.folder.relative_to(run)}    （summary.json / episodes.csv / seed_*.npz）"], merge=True)
        row([f"权重 {s['checkpoint']}    SHA-256 {s['checkpoint_sha256']}"], merge=True)
    row(["逐回合验收  |  机间距离取三对中的最小三维中心距。墙面净距已扣除机体半径。"], "A1", 30, True)
    row(["评估集", "训练组", "智能体交互", "种子", "控制步", "执行结局", "严格结局", "首次机间接触", "飞行中机间终止",
         "首次接触控制步", "首次接触子步", "起点间距（cm）", "最近机间距（cm）", "墙面最小净距（cm）", "首次接触对象"], "A4", 35)
    count_episodes = 0
    for (split, arm, step), cohort in cohorts.items():
        for seed, episode in sorted(cohort.records.items()):
            row([SPLIT[split], ARM[arm], step, seed, episode["control_steps"],
                 EXECUTION.get(episode["execution_outcome"], episode["execution_outcome"]), OUTCOME[episode["strict_outcome"]],
                 episode["peer_first"], episode["common_peer_stop"], episode["first_control_step"], episode["first_substep"],
                 round(100*episode["initial_min_peer_m"], 4), round(100*episode["min_peer_center_m"], 4),
                 round(100*episode["min_wall_surface_clearance_m"], 4), episode["first_objects"] or "无"])
            sheet.merge_cells(start_row=line, start_column=15, end_row=line, end_column=17)
            count_episodes += 1
    return checks, count_episodes


def sheet_part(archive, sheet_element):
    rid = sheet_element.get(f"{{{OFFICE}}}id")
    rel = next(r for r in et.fromstring(archive.read(REL)) if r.get("Id") == rid)
    target = rel.get("Target")
    return target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))


def graft(source, draft, destination):
    """Only the three sheet registries change. Existing worksheet/style ZIP payloads are retained."""
    with ZipFile(source) as original, ZipFile(draft) as generated, ZipFile(destination, "w") as output:
        wb, rels, types = (et.fromstring(original.read(name)) for name in (WB, REL, CT))
        sheets = wb.find(f"{{{MAIN}}}sheets")
        number = max(int(node.get("sheetId")) for node in sheets) + 1
        while f"xl/worksheets/sheet{number}.xml" in original.namelist():
            number += 1
        part = f"xl/worksheets/sheet{number}.xml"
        used = {r.get("Id") for r in rels}
        rid_number = 1
        while f"rId{rid_number}" in used:
            rid_number += 1
        rid = f"rId{rid_number}"
        et.SubElement(sheets, f"{{{MAIN}}}sheet", name=SHEET, sheetId=str(number), attrib={f"{{{OFFICE}}}id": rid})
        et.SubElement(rels, f"{{{PACKAGE}}}Relationship", Id=rid, Type=OFFICE+"/worksheet", Target="/"+part)
        et.SubElement(types, f"{{{TYPES}}}Override", PartName="/"+part,
                      ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
        registries = {WB: wb, REL: rels, CT: types}
        for member in original.infolist():
            data = (et.tostring(registries[member.filename], encoding="utf-8", xml_declaration=True)
                    if member.filename in registries else original.read(member))
            output.writestr(member, data)
        draft_sheets = et.fromstring(generated.read(WB)).find(f"{{{MAIN}}}sheets")
        new_sheet = next(s for s in draft_sheets if s.get("name") == SHEET)
        output.writestr(part, generated.read(sheet_part(generated, new_sheet)), compress_type=ZIP_DEFLATED)
    return part


def export(run, source, output):
    if source.resolve() == output.resolve() or output.suffix.lower() != ".xlsx":
        raise ValueError("Specify a separate .xlsx output to preserve the source")
    spec, best, cohorts = load_experiment(run)
    source_hash = hashlib.sha256(source.read_bytes()).digest()
    book = load_workbook(source)
    old_names = book.sheetnames
    if SHEET in old_names:
        raise ValueError(f"The source workbook already contains {SHEET}")
    checks, episode_count = add_panel(book, run, spec, best, cohorts)
    expected_rows = book[SHEET].max_row
    if episode_count != sum(c.summary["episodes"] for c in cohorts.values()):
        raise ValueError("Workbook is missing episode rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        draft, result = Path(temp)/"draft.xlsx", Path(temp)/"result.xlsx"
        book.save(draft)
        book.close()
        part = graft(source, draft, result)
        with ZipFile(source) as old, ZipFile(result) as new:
            members = set(old.namelist())
            if set(new.namelist()) != members | {part} or new.testzip():
                raise ValueError("Invalid exported workbook package")
            for item in members - {WB, REL, CT}:
                if new.read(item) != old.read(item):
                    raise ValueError(f"Source workbook part changed: {item}")
        verified = load_workbook(result, data_only=True, read_only=True)
        try:
            if verified.sheetnames != old_names + [SHEET] or verified[SHEET].max_row != expected_rows:
                raise ValueError("Invalid worksheet layout")
            for r, value in checks.items():
                if verified[SHEET].cell(r, 6).value != value:
                    raise ValueError(f"Incorrect measured counts in row {r}")
        finally:
            verified.close()
        if hashlib.sha256(source.read_bytes()).digest() != source_hash:
            raise RuntimeError("Source workbook changed during export; rerun with the new version")
        try:
            os.link(result, output)
        except FileExistsError:
            with ZipFile(output) as saved, ZipFile(result) as new:
                if set(saved.namelist()) != set(new.namelist()) or any(saved.read(p) != new.read(p) for p in new.namelist()):
                    raise FileExistsError(f"Choose a separate file; {output} already contains different data") from None
    print(f"{output} · {len(cohorts)} 组评估，{episode_count} 条逐回合记录")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source_workbook", type=Path)
    parser.add_argument("new_workbook", type=Path)
    args = parser.parse_args()
    export(args.run.resolve(), args.source_workbook.resolve(), args.new_workbook.resolve())


if __name__ == "__main__":
    main()
