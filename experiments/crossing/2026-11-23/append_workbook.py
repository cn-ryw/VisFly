"""Append measured results while preserving all earlier XLSX worksheet and style bytes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import posixpath
import shutil
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from lxml import etree as et
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

PANEL = "论文同期三机基线"
OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PACKAGE = "http://schemas.openxmlformats.org/package/2006/relationships"
TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def json_data(p):
    return json.loads(p.read_text())


def add_panel(book, run):
    sheet = book.create_sheet(PANEL)
    old = book["墙前纠偏实验"]
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "B5"
    for i,w in enumerate([19,18,15,15,13,12,12,12,13,17,14,15],1):
        sheet.column_dimensions[get_column_letter(i)].width = w

    def row(index, cells, template="A2", height=24):
        for col, val in enumerate(cells,1):
            c = sheet.cell(index,col,val)
            c._style = old[template]._style
        sheet.row_dimensions[index].height = height

    def title(index, value, template="A2", height=24):
        sheet.merge_cells(start_row=index,start_column=1,end_row=index,end_column=12)
        row(index,[value], template,height)

    title(1,"VisFly 论文同期基线  |  车库三机安全穿缝","A1",36)
    title(2,"训练种子 42，24 个车库各 3 架，PPO 连续训练。每个种子运行一个团队回合。每行的分母均为 100 个团队。")
    title(3,"原生为作者环境终止标志。旧验收使用控制帧。严验按 2.5 ms 子步检查墙与机库，并要求机间中心距至少 20 cm。")
    row(4,["评估集","训练交互","团队回合","原生通过","旧验收通过","严验通过","严验碰撞","严验超时","严验绕行","严验未完成","平均控制步","训练秒数"],"A4",32)
    protocol = json_data(run/"setup/protocol.json")
    cohorts = [run/f"evaluation/development/PPO_{s}" for s in protocol["post_update_checkpoints"]]
    cohorts.append(run/"evaluation/held-out/PPO_10008576")
    results = [json_data(c/"summary.json") for c in cohorts]
    for i,v in enumerate(results,5):
        strict = v["counts"]["strict"]
        st = v["agent_steps"]
        seconds = json_data(run/f"train/PPO_{st}.json")["elapsed_s"]
        row(i,["开发 1000–1099" if v["split"]=="development" else "留出 20000–20099",st,v["episodes"],
               v["counts"]["native"].get("success",0),v["counts"]["legacy"].get("success",0),
               *[strict.get(k,0) for k in ("success","collision","timeout","bypass","incomplete")],
               round(v["mean_control_steps"],2),round(seconds,1)])
    title(11,"图 7 原生训练成功率  |  图像数字化，两个窗口分别按交互数积分","A4",29)
    row(12,["开始交互","结束交互","论文均值 %","容许下限 %","本机均值 %","差值/百分点","达标"],"A4",32)
    curves = json_data(run/"figures/curve-windows.json")
    for i,v in enumerate(curves["windows"],13):
        row(i,[v["low"],v["high"],round(v["paper"]*100,3),round(v["required"]*100,3),
               round(v["actual"]*100,3),round(v["difference_points"],3),"是" if v["meets_threshold"] else "否"])
    title(16,"历史参照  |  11 月 18 日修正环境，1,000,704 交互，种子 1000–1099，旧控制帧验收 4/100。")
    title(17,"留出验收线为严验通过至少 80/100。未完成指原生终止时，几何未测到碰撞，且三机未完成真实穿缝并越线。")
    title(18,"实验归档  "+str(run))
    title(19,"原生训练逐轮  train/training.csv    终端日志  logs/train.log    逐回合  evaluation/<评估集>/PPO_<交互数>/episodes.csv")
    for i,st in enumerate(protocol["post_update_checkpoints"],20):
        meta=json_data(run/f"train/PPO_{st}.json")
        title(i,f"train/PPO_{st}.zip    SHA-256  {meta['sha256']}")
    title(25,"逐回合实测  |  首碰以积分区间右端记步，初始接触为 0。墙净空计入 10 cm 机体半径。","A4",29)
    row(26,["评估集","训练交互","种子","控制步","原生结局","旧验收结局","严验结局","首碰控制步","步内子步","首碰对象","机间最小 m","墙净空 m"],"A4",32)
    labels={"success":"通过","collision":"碰撞","timeout":"超时","bypass":"绕行","incomplete":"未完成"}
    line=27
    for cohort in cohorts:
        for r in csv.DictReader((cohort/"episodes.csv").open()):
            row(line,["开发" if cohort.parent.name=="development" else "留出",int(cohort.name.split("_")[1]),int(r["seed"]),
                      int(r["control_steps"]),*[labels[r[key+"_outcome"]] for key in ("native","legacy","strict")],
                      *[int(r[k]) if r[k] else None for k in ("first_contact_step","first_contact_substep")],
                      r["first_contact_objects"] or None,round(float(r["min_peer_center_m"]),5),
                      round(float(r["min_gap_wall_surface_clearance_m"]),5)],height=19)
            line+=1
    assert line==527
    return results


def graft(source, edited, output):
    with ZipFile(source) as old, ZipFile(edited) as draft, ZipFile(output,"w") as result:
        item = max(int(n.split("sheet")[-1].split(".")[0]) for n in old.namelist()
                   if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))+1
        name=f"xl/worksheets/sheet{item}.xml"
        wb=et.fromstring(old.read("xl/workbook.xml"))
        relationships=et.fromstring(old.read("xl/_rels/workbook.xml.rels"))
        types=et.fromstring(old.read("[Content_Types].xml"))
        sheets=wb.find(f"{{{MAIN}}}sheets")
        rid="rIdVisFlyPaperCrossing"
        assert rid not in {node.get("Id") for node in relationships}
        sheetid=max(int(s.get("sheetId")) for s in sheets)+1
        et.SubElement(sheets,f"{{{MAIN}}}sheet",name=PANEL,sheetId=str(sheetid),attrib={f"{{{OFFICE}}}id":rid})
        et.SubElement(relationships,f"{{{PACKAGE}}}Relationship",Id=rid,Type=OFFICE+"/worksheet",Target="/"+name)
        et.SubElement(types,f"{{{TYPES}}}Override",PartName="/"+name,
                      ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
        changed={"xl/workbook.xml":wb,"xl/_rels/workbook.xml.rels":relationships,"[Content_Types].xml":types}
        for entry in old.infolist():
            if entry.filename in changed:
                data=et.tostring(changed[entry.filename],encoding="utf-8",xml_declaration=True)
            else:
                data=old.read(entry)
            result.writestr(entry,data)
        # openpyxl writes inline strings. Styles on this sheet are all from the original book.
        generated=et.fromstring(draft.read("xl/workbook.xml")).find(f"{{{MAIN}}}sheets")[-1]
        draft_rid=generated.get(f"{{{OFFICE}}}id")
        rel=next(s for s in et.fromstring(draft.read("xl/_rels/workbook.xml.rels")) if s.get("Id")==draft_rid)
        target=rel.get("Target")
        path=target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl",target))
        result.writestr(name,draft.read(path),compress_type=8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("workbook",type=Path)
    args=parser.parse_args()
    run,target=args.run.resolve(),args.workbook.resolve()
    evidence=run/"workbook"
    evidence.mkdir(exist_ok=True)
    marker=evidence/"installation.json"
    if marker.exists() and json_data(marker)["after_sha256"]==digest(target):
        return
    start=digest(target)
    backup=evidence/f"before-{start[:12]}.xlsx"
    shutil.copy2(target,backup)
    book=load_workbook(backup)
    original_names=book.sheetnames.copy()
    if PANEL in original_names:
        raise RuntimeError("An existing different baseline sheet needs review")
    old_formula_count=sum(c.data_type=="f" for s in book for row in s for c in row)
    results=add_panel(book,run)
    with NamedTemporaryFile(suffix=".xlsx",dir=evidence) as intermediate, NamedTemporaryFile(
            suffix=".xlsx",dir=target.parent,delete=False) as pending:
        tmp=Path(pending.name)
        try:
            book.save(intermediate.name)
            graft(backup,intermediate.name,tmp)
            with ZipFile(backup) as a, ZipFile(tmp) as b:
                assert b.testzip() is None
                for name in a.namelist():
                    if name not in {"xl/workbook.xml","xl/_rels/workbook.xml.rels","[Content_Types].xml"}:
                        assert a.read(name)==b.read(name),name
            test=load_workbook(tmp)
            assert test.sheetnames==original_names+[PANEL]
            s=test[PANEL]
            assert s.max_row==526 and s["F9"].value==results[-1]["counts"]["strict"].get("success",0)
            assert s["C526"].value==20099
            for i,r in enumerate(results,5):
                assert sum(s.cell(i,c).value for c in range(6,11))==r["episodes"]==100
            for old in original_names:
                assert all(test[old][cell.coordinate].value==cell.value for row in book[old] for cell in row)
            assert digest(target)==start, "Workbook changed during preparation"
            shutil.copymode(target,tmp)
            os.replace(tmp,target)
            marker.write_text(json.dumps(dict(path=str(target),before_sha256=start,after_sha256=digest(target),
                  historical_sheets=original_names,unchanged_formulas=old_formula_count,appended_episodes=500),
                  ensure_ascii=False,indent=2)+"\n")
        finally:
            tmp.unlink(missing_ok=True)


if __name__=="__main__":
    main()
