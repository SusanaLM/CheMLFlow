"""Self-contained Plotly and vanilla-JavaScript EDA report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..schema import is_log_activity_property


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def _records(frame: pd.DataFrame, limit: Optional[int] = None) -> list[dict]:
    selected = frame if limit is None else frame.head(limit)
    return _clean_json(selected.to_dict(orient="records"))


def write_html_report(
    path: Path,
    *,
    profile: Dict[str, Any],
    molecule_table: pd.DataFrame,
    descriptor_summary: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    nearest_neighbors: pd.DataFrame,
    activity_cliffs: pd.DataFrame,
    galleries: Dict[str, list[int]],
    primary_property: Optional[str],
    property_profile: Optional[Dict[str, Any]],
    dataset_health: Optional[Dict[str, Any]],
    dataset_warnings: list[Dict[str, str]],
    property_distribution: Dict[str, Any],
    druglikeness_summary: Optional[Dict[str, Any]],
    structural_alerts: pd.DataFrame,
    model_readiness: Optional[Dict[str, Any]],
    advanced: bool,
    drug_panel_enabled: bool,
    use_scattergl: bool,
    map_method: str,
    top_scaffolds: int,
    max_points_for_svg_hover: int = 5000,
    selection_columns: Optional[list[str]] = None,
) -> None:
    """Write an offline report bundle with linked chemical interpretation views."""
    try:
        from plotly.offline import get_plotlyjs
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The optional EDA report requires Plotly. Install the package with "
            "`pip install plotly` or install the project dependencies."
        ) from exc

    payload = {
        "profile": profile,
        "molecules": _records(molecule_table),
        "descriptor_summary": _records(descriptor_summary),
        "scaffolds": _records(scaffold_summary, top_scaffolds),
        "scaffolds_high": _records(
            scaffold_summary.sort_values(
                "property_median", ascending=False, na_position="last"
            )
            if "property_median" in scaffold_summary.columns
            else scaffold_summary.iloc[0:0],
            top_scaffolds,
        ),
        "scaffolds_low": _records(
            scaffold_summary.sort_values(
                "property_median", ascending=True, na_position="last"
            )
            if "property_median" in scaffold_summary.columns
            else scaffold_summary.iloc[0:0],
            top_scaffolds,
        ),
        "clusters": _records(cluster_summary, top_scaffolds),
        "neighbors": _records(nearest_neighbors),
        "cliffs": _records(activity_cliffs, 250),
        "galleries": galleries,
        "property": primary_property,
        "property_profile": property_profile,
        "dataset_health": dataset_health,
        "dataset_warnings": dataset_warnings,
        "property_distribution": property_distribution,
        "druglikeness_summary": druglikeness_summary,
        "structural_alerts": _records(structural_alerts, 100),
        "model_readiness": model_readiness,
        "advanced": advanced,
        "drug_panel_enabled": drug_panel_enabled,
        "use_scattergl": use_scattergl,
        "is_log_activity": bool(
            primary_property and is_log_activity_property(primary_property)
        ),
        "map_method": map_method,
        "max_points_for_svg_hover": int(max_points_for_svg_hover),
        "selection_columns": list(selection_columns or ["compound_id"]),
    }
    payload_json = json.dumps(_clean_json(payload), allow_nan=False).replace("</", "<\\/")
    plotly_js = get_plotlyjs()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Molecular Landscape EDA Report</title>
<style>
:root {{ --ink:#17202a; --muted:#52616b; --accent:#1261a0; --line:#dbe3e8;
  --panel:#ffffff; --soft:#f4f7f9; --warn:#fff4d6; }}
* {{ box-sizing:border-box; }}
html {{ font-size:21px; }}   /* ~30% larger base; scales all rem/em text */
body {{ margin:0; font-family:Arial, Helvetica, sans-serif; color:var(--ink);
  background:var(--soft); line-height:1.45; }}
header {{ background:linear-gradient(120deg,#102a43,#1261a0); color:white; padding:30px 5vw; }}
header p {{ max-width:1000px; color:#d8ebfa; }}
main {{ max-width:1500px; margin:auto; padding:24px; }}
section {{ background:var(--panel); margin:18px 0; padding:24px; border:1px solid var(--line);
  border-radius:12px; box-shadow:0 4px 18px #17324d0d; }}
h1,h2,h3 {{ line-height:1.18; }}
.grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); }}
.metric {{ padding:14px; border-left:4px solid var(--accent); background:var(--soft); }}
.metric b {{ display:block; font-size:1.55rem; }}
.warning {{ background:var(--warn); border-left:4px solid #d99a00; padding:10px 14px; margin:8px 0; }}
.critical {{ background:#fde7e7; border-left-color:#a61b1b; }}
.ok {{ background:#eaf6ed; border-left:4px solid #2b7a3d; padding:10px 14px; margin:8px 0; }}
.note {{ color:var(--muted); }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:12px; }}
.card {{ border:1px solid var(--line); border-radius:9px; padding:10px; overflow:hidden; }}
.card img {{ width:100%; height:150px; object-fit:contain; background:white; }}
.card code {{ display:block; overflow-wrap:anywhere; font-size:.72rem; }}
.plot {{ min-height:380px; }}
.map-layout {{ display:grid; grid-template-columns:minmax(0,3fr) minmax(290px,1fr); gap:16px; }}
#structure-map {{ min-height:680px; }}
#molecule-panel {{ border:1px solid var(--line); border-radius:9px; padding:14px; overflow:auto; }}
#molecule-panel img {{ width:100%; max-height:250px; object-fit:contain; }}
#selected-ids {{ width:100%; min-height:90px; font-family:ui-monospace,monospace; }}
table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:7px; vertical-align:top; }}
button,select {{ font-family:inherit; font-size:1rem; border:1px solid #9fb3c8; border-radius:5px; background:white; padding:7px 10px; cursor:pointer; }}
.cliff {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }}
.cliff img {{ width:100%; height:160px; object-fit:contain; }}
.small {{ font-size:.82rem; color:var(--muted); }}
@media(max-width:850px) {{ .map-layout {{ grid-template-columns:1fr; }} }}
</style>
<script>{plotly_js}</script>
</head>
<body>
<header>
<h1>Molecular Landscape Exploratory Data Analysis</h1>
<p>This report connects molecular structures, descriptors, scaffolds, local Tanimoto
neighborhoods, and selected properties. It is exploratory evidence, not proof of a
mechanistic structure-activity relationship. It describes the post-curation molecular
cohort supplied by the pipeline, not the original assay-record dataset.</p>
</header>
<main>
<section id="overview"><h2>1. Executive dataset summary</h2>
<div id="overview-metrics" class="grid"></div><div id="schema"></div>
<div id="executive-notes"></div></section>

<section><h2>2. Dataset health and warnings</h2>
<div id="health-summary"></div><div id="warnings"></div></section>

<section><h2>3. Property profile</h2>
<div id="property-profile"></div></section>

<section id="gallery-section"><h2>4. Molecule gallery</h2>
<p class="note">Representative galleries are deterministic. A missing depiction indicates
the configured SVG limit or an isolated depiction failure.</p>
<div id="galleries"></div></section>

<section><h2>5. Molecular descriptor distributions</h2>
<div id="descriptor-distributions" class="plot"></div>
<p class="note">Descriptor outliers are robust univariate flags for review, not automatic exclusions.</p></section>

<section id="drug-discovery-section"><h2>6. Drug-discovery heuristics</h2>
<div class="warning">These are small-molecule drug-discovery heuristics, not universal
filters. They may not apply to macrocycles, peptides, PROTACs, covalent fragments,
materials, QM datasets, or non-oral concepts.</div>
<div id="drug-summary" class="grid"></div><div id="qed-distribution" class="plot"></div>
<div id="lipinski-distribution" class="plot"></div><div id="alert-table"></div></section>

<section><h2>7. Property distribution</h2>
<div id="property-notes"></div><div id="property-histogram" class="plot"></div>
<div id="class-chart" class="plot"></div><div id="property-descriptor" class="plot"></div></section>

<section><h2>8. Interactive molecular maps</h2>
<div class="warning"><b>Scientific contract:</b> the coordinates shown here are
structure-only by default. Colour may show a property without affecting point positions.
The optional property-aware geometry is supervised and must not be presented as independent
SAR evidence. Property-aware maps are supervised. PCA and UMAP do not prove mechanism.</div>
<label>Geometry <select id="geometry-select"></select></label>
<label>Colour points by <select id="colour-select"></select></label>
<span id="selection-summary" class="small">No points selected.</span>
<button id="download-selection" type="button">Download selected IDs as CSV</button>
<div class="map-layout"><div id="structure-map"></div><aside id="molecule-panel">
Hover or click a point to inspect its molecule and nearest neighbors.</aside></div></section>

<section><h2>9. Scaffold and chemical-series browser</h2>
<p class="note">Scaffolds are Bemis-Murcko families; clusters are Butina clusters at the
configured similarity operating point. Neither is a universal chemical truth.</p>
<h3>Dominant scaffolds</h3><div id="scaffold-browser"></div>
<h3>Scaffolds with high and low median property</h3><div id="scaffold-high-low"></div>
<h3>Dominant Butina clusters</h3><div id="cluster-browser"></div></section>

<section><h2>10. Nearest neighbours and activity cliffs/local discontinuities (similarity-defined screen)</h2>
<p id="cliff-explanation" class="note"></p><div id="cliff-browser" class="grid"></div></section>

<section><h2>11. Model-readiness and next-step recommendations</h2>
<div id="model-readiness"></div></section>

<section><h2>12. Export and reproducibility</h2>
<p>Use lasso or box selection on the structure-only map to identify compounds of interest.
Selected IDs are shown below and can be downloaded as CSV. All finalized artifacts,
configuration, dependency versions, and checksums are recorded with this run.</p>
<textarea id="selected-ids" readonly placeholder="No compounds selected"></textarea>
<div id="reproducibility"></div></section>
</main>
<script>
const DATA = {payload_json};
const byIndex = new Map(DATA.molecules.map(x => [Number(x.structure_index), x]));
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch =>
  ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[ch]));
const fmt = value => value === null || value === undefined || Number.isNaN(value)
  ? "NA" : (typeof value === "number" ? value.toFixed(3) : String(value));
const image = row => row && row.svg_path
  ? `<img loading="lazy" src="${{esc(row.svg_path)}}" alt="Molecule depiction">`
  : `<div class="note">No SVG depiction available</div>`;
const prop = DATA.property;
const metrics = DATA.profile.counts;
document.getElementById("overview-metrics").innerHTML = [
 ["Input rows",metrics.input_rows],["Valid molecules",metrics.valid_molecules],
 ["Invalid molecules",metrics.invalid_molecules],["Scaffolds",metrics.scaffolds],
 ["Singleton scaffolds",metrics.singleton_scaffolds],["Clusters",metrics.clusters],
 ["Descriptor outliers",metrics.descriptor_outlier_molecules],["Local discontinuities",DATA.cliffs.length]
].map(x=>`<div class="metric"><span>${{esc(x[0])}}</span><b>${{esc(x[1]??"NA")}}</b></div>`).join("");
const schema=DATA.profile.schema;
document.getElementById("schema").innerHTML=`<p><b>SMILES:</b> ${{esc(schema.smiles_column)}} &nbsp;
<b>ID:</b> ${{esc(schema.id_column)}} &nbsp; <b>Selected property:</b> ${{esc((schema.property_columns||[]).join(", ")||"none")}}</p>`;
document.getElementById("executive-notes").innerHTML=`<ul>${{DATA.profile.interpretation_notes.map(x=>`<li>${{esc(x)}}</li>`).join("")}}</ul>`;
const structuredWarnings=DATA.dataset_warnings||[];
const warningMessages=structuredWarnings.length?structuredWarnings.map(x=>x.message):DATA.profile.warnings;
document.getElementById("warnings").innerHTML=(warningMessages.length
 ? warningMessages.map(x=>`<div class="warning">${{esc(x)}}</div>`).join("")
 : `<p>No automated warning rules fired.</p>`);
document.getElementById("health-summary").innerHTML=DATA.dataset_health
 ? `<p>${{esc(DATA.dataset_health.plain_language_summary)}}</p><div class="grid">
 <div class="metric"><span>Duplicate ID rows</span><b>${{DATA.dataset_health.duplicate_id_rows}}</b></div>
 <div class="metric"><span>Duplicate structure rows</span><b>${{DATA.dataset_health.duplicate_canonical_structure_rows}}</b></div>
 <div class="metric"><span>Dot-disconnected SMILES</span><b>${{DATA.dataset_health.dot_disconnected_smiles_count}}</b></div>
 <div class="metric"><span>Charged fraction</span><b>${{fmt(DATA.dataset_health.charge_distribution.charged_fraction)}}</b></div></div>`
 : `<p class="note">Run with <code>--advanced-eda</code> for expanded chemistry-health diagnostics.</p>`;
const pp=DATA.property_profile;
document.getElementById("property-profile").innerHTML=pp
 ? `<div class="grid"><div class="metric"><span>Property</span><b>${{esc(pp.display_name)}}</b></div>
 <div class="metric"><span>Semantic type</span><b>${{esc(pp.semantic_type)}}</b></div>
 <div class="metric"><span>Units</span><b>${{esc(pp.units??"not established")}}</b></div>
 <div class="metric"><span>Direction</span><b>${{pp.higher_is_better===null?"context dependent":(pp.higher_is_better?"higher is favourable":"lower is favourable")}}</b></div></div>
 <p>${{pp.is_potency?"Recognised as potency-like; potency-specific language is used cautiously.":"Neutral property language is used."}}</p>`
 : `<p>No property was selected.</p>`;

function card(row) {{
 return `<div class="card">${{image(row)}}<b>${{esc(row.compound_id)}}</b>
 <div class="small">${{prop ? `${{esc(prop)}}: ${{fmt(row[prop])}}` : "No property selected"}}</div>
 <div class="small">Scaffold ${{esc(row.scaffold_id)}}; cluster ${{esc(row.cluster_id)}}</div>
 <code>${{esc(row.canonical_smiles)}}</code></div>`;
}}
const galleryNames={{random_representatives:"Random representatives",
 top_scaffold_representatives:"Top scaffold representatives",high_property:"High-property molecules",
 low_property:"Low-property molecules",descriptor_outliers:"Descriptor outliers",
 structural_outliers:"Structural outliers (singleton scaffold and cluster)",
 high_qed:"High-QED molecules",low_qed:"Low-QED molecules",
 many_lipinski_violations:"Molecules with many Lipinski violations"}};
document.getElementById("galleries").innerHTML=Object.entries(DATA.galleries).map(([name,ids]) =>
 `<h3>${{esc(galleryNames[name]||name)}}</h3><div class="gallery">${{ids.slice(0,48).map(id=>card(byIndex.get(Number(id)))).join("")}}</div>`).join("");

const config={{responsive:true,displaylogo:false}};
const FONT={{family:"Arial, Helvetica, sans-serif",size:16}};
const plot=(div,data,layout,cfg)=>Plotly["newPlot"](div,data,Object.assign({{font:FONT}},layout||{{}}),cfg);
const numeric = key => DATA.molecules.map(x=>x[key]).filter(x=>x!==null && x!==undefined && Number.isFinite(Number(x))).map(Number);
const propertyNotes=(DATA.property_distribution&&DATA.property_distribution.interpretation_notes)||[];
document.getElementById("property-notes").innerHTML=`<ul>${{propertyNotes.map(x=>`<li>${{esc(x)}}</li>`).join("")}}</ul>`;
if(prop && DATA.property_distribution.kind==="numeric") plot("property-histogram",[{{x:numeric(prop),type:"histogram",marker:{{color:"#1261a0"}}}}],
 {{title:`Distribution of ${{prop}}`,xaxis:{{title:prop}},yaxis:{{title:"Molecules"}},margin:{{t:55}}}},config);
else document.getElementById("property-histogram").innerHTML="<p>No property was selected.</p>";
const classCounts=DATA.property_distribution.class_counts||DATA.profile.activity_class_counts||{{}};
if(Object.keys(classCounts).length) plot("class-chart",[{{x:Object.keys(classCounts),y:Object.values(classCounts),type:"bar",marker:{{color:"#5a8f29"}}}}],
 {{title:"Activity/class counts",yaxis:{{title:"Molecules"}},margin:{{t:55}}}},config);
else document.getElementById("class-chart").style.display="none";
const basicDescriptors=["MolWt","LogP","TPSA","HBD","HBA","RotBonds","RingCount"];
const advancedDescriptors=["MolWt","MolLogP","TPSA","NumHDonors","NumHAcceptors","NumRotatableBonds","RingCount","FractionCSP3","FormalCharge"];
const descriptorKeys=DATA.advanced?advancedDescriptors:basicDescriptors;
plot("descriptor-distributions",descriptorKeys.map(d=>({{x:numeric(d),type:"histogram",name:d,opacity:.55}})),
 {{title:"Descriptor distributions",barmode:"overlay",xaxis:{{title:"Descriptor value"}},yaxis:{{title:"Molecules"}},margin:{{t:55}}}},config);
if(prop && DATA.property_distribution.kind==="numeric") plot("property-descriptor",descriptorKeys.map(d=>({{x:numericAligned(d,prop),y:numericAligned(prop,d),mode:"markers",type:DATA.use_scattergl?"scattergl":"scatter",name:d,marker:{{size:5,opacity:.55}}}})),
 {{title:`${{prop}} versus descriptors`,xaxis:{{title:"Descriptor value"}},yaxis:{{title:prop}},margin:{{t:55}}}},config);
else document.getElementById("property-descriptor").style.display="none";
function numericAligned(key, requireKey) {{
 const valid=v=>v!==null&&v!==undefined&&Number.isFinite(Number(v));
 return DATA.molecules.filter(x=>valid(x[key]) && (!requireKey || valid(x[requireKey]))).map(x=>Number(x[key]));
}}
if(DATA.drug_panel_enabled && DATA.druglikeness_summary) {{
 const ds=DATA.druglikeness_summary;
 document.getElementById("drug-summary").innerHTML=[
  ["Lipinski pass fraction",fmt(ds.lipinski_pass_fraction)],
  ["Median QED",fmt(ds.qed.median)],["Lead-like",ds.lead_like_count],
  ["Fragment-like",ds.fragment_like_count],["Structural alerts",ds.structural_alert_records]
 ].map(x=>`<div class="metric"><span>${{esc(x[0])}}</span><b>${{esc(x[1])}}</b></div>`).join("");
 plot("qed-distribution",[{{x:numeric("QED"),type:"histogram",marker:{{color:"#5a8f29"}}}}],
  {{title:"QED distribution",xaxis:{{title:"QED"}},yaxis:{{title:"Molecules"}},margin:{{t:55}}}},config);
 plot("lipinski-distribution",[{{x:numeric("Lipinski_Violation_Count"),type:"histogram",marker:{{color:"#b65d20"}}}}],
  {{title:"Lipinski violation-count distribution",xaxis:{{title:"Violation count"}},yaxis:{{title:"Molecules"}},margin:{{t:55}}}},config);
 document.getElementById("alert-table").innerHTML=DATA.structural_alerts.length
  ? `<h3>Structural alerts (first 100 records)</h3><table><tr><th>Compound</th><th>Alert</th></tr>${{DATA.structural_alerts.map(a=>`<tr><td>${{esc(a.compound_id)}}</td><td>${{esc(a.alert)}}</td></tr>`).join("")}}</table>`
  : `<p class="note">No structural alert records were generated, or the optional catalogue was unavailable.</p>`;
}} else document.getElementById("drug-discovery-section").style.display="none";

let xKey=`structure_${{DATA.map_method}}_x`, yKey=`structure_${{DATA.map_method}}_y`;
const propertyX=`property_aware_${{DATA.map_method}}_x`, propertyY=`property_aware_${{DATA.map_method}}_y`;
const hasPropertyGeometry=DATA.molecules.some(m=>m[propertyX]!==null&&m[propertyX]!==undefined&&m[propertyY]!==null&&m[propertyY]!==undefined&&Number.isFinite(Number(m[propertyX]))&&Number.isFinite(Number(m[propertyY])));
document.getElementById("geometry-select").innerHTML=`<option value="structure">Structure-only (least circular)</option>${{hasPropertyGeometry?'<option value="property">Property-aware (supervised)</option>':""}}`;
const colourOptions=[prop,"property_bin","activity_class","scaffold_id","cluster_id","MolWt","MolLogP","LogP","TPSA","QED","Lipinski_Violation_Count"].filter((x,i,a)=>x&&a.indexOf(x)===i&&DATA.molecules.some(m=>m[x]!==null&&m[x]!==undefined));
document.getElementById("colour-select").innerHTML=colourOptions.map(x=>`<option value="${{esc(x)}}">${{esc(x)}}</option>`).join("");
function colourSpec(key) {{
 const raw=DATA.molecules.map(x=>x[key]);
 if(raw.every(x=>x===null||x===undefined||Number.isFinite(Number(x)))) return {{values:raw.map(x=>x===null||x===undefined?null:Number(x)),categorical:false}};
 const cats=[...new Set(raw.map(x=>String(x??"missing")))].sort(); const codes=new Map(cats.map((x,i)=>[x,i]));
 return {{values:raw.map(x=>codes.get(String(x??"missing"))),categorical:true,categories:cats}};
}}
const hover=DATA.molecules.map((m,i)=>i<DATA.max_points_for_svg_hover?`<b>${{esc(m.compound_id)}}</b><br>${{prop?`${{esc(prop)}}: ${{fmt(m[prop])}}<br>`:""}}Scaffold: ${{m.scaffold_id}}<br>Cluster: ${{m.cluster_id}}<br>MolWt: ${{fmt(m.MolWt)}}<br>LogP: ${{fmt(m.MolLogP??m.LogP)}}<br>TPSA: ${{fmt(m.TPSA)}}<br>HBD/HBA: ${{fmt(m.NumHDonors??m.HBD)}} / ${{fmt(m.NumHAcceptors??m.HBA)}}<br>RotBonds/Rings: ${{fmt(m.NumRotatableBonds??m.RotBonds)}} / ${{fmt(m.RingCount)}}<br>QED: ${{fmt(m.QED)}}`:`<b>${{esc(m.compound_id)}}</b>`);
let initialColour=colourSpec(colourOptions[0]);
const mapTrace={{x:DATA.molecules.map(x=>x[xKey]),y:DATA.molecules.map(x=>x[yKey]),mode:"markers",type:DATA.use_scattergl?"scattergl":"scatter",
 text:hover,hovertemplate:"%{{text}}<extra></extra>",customdata:DATA.molecules.map(x=>x.structure_index),
 marker:{{size:8,opacity:.8,color:initialColour.values,colorscale:"Viridis",showscale:true,colorbar:{{title:colourOptions[0],tickvals:initialColour.categorical?initialColour.categories.map((_,i)=>i):undefined,ticktext:initialColour.categorical?initialColour.categories:undefined}}}}}};
plot("structure-map",[mapTrace],{{title:`Structure-only ${{DATA.map_method.toUpperCase()}} map`,
 dragmode:"lasso",xaxis:{{title:`${{DATA.map_method.toUpperCase()}} 1`}},yaxis:{{title:`${{DATA.map_method.toUpperCase()}} 2`}},margin:{{t:55}}}},config);
document.getElementById("colour-select").addEventListener("change",e=>{{const spec=colourSpec(e.target.value); Plotly.restyle("structure-map",
 {{"marker.color":[spec.values],"marker.colorbar.title":e.target.value,"marker.colorbar.tickvals":[spec.categorical?spec.categories.map((_,i)=>i):null],"marker.colorbar.ticktext":[spec.categorical?spec.categories:null]}});}});
document.getElementById("geometry-select").addEventListener("change",e=>{{
 const supervised=e.target.value==="property";
 xKey=supervised?propertyX:`structure_${{DATA.map_method}}_x`;
 yKey=supervised?propertyY:`structure_${{DATA.map_method}}_y`;
 Plotly.update("structure-map",{{x:[DATA.molecules.map(m=>m[xKey])],y:[DATA.molecules.map(m=>m[yKey])]}},
  {{title:supervised?`Property-aware ${{DATA.map_method.toUpperCase()}} map (supervised)`:`Structure-only ${{DATA.map_method.toUpperCase()}} map`}});
}});
const map=document.getElementById("structure-map");
map.on("plotly_hover",e=>showMolecule(Number(e.points[0].customdata)));
map.on("plotly_click",e=>showMolecule(Number(e.points[0].customdata)));
let selectedIndices=[];
map.on("plotly_selected",e=>{{
 selectedIndices=e?e.points.map(p=>Number(p.customdata)):[];
 const selectedIds=selectedIndices.map(i=>byIndex.get(i).compound_id);
 document.getElementById("selection-summary").textContent=selectedIndices.length?`${{selectedIndices.length}} points selected.`:"No points selected.";
 document.getElementById("selected-ids").value=selectedIds.join("\\n");
}});
document.getElementById("download-selection").addEventListener("click",()=>{{
 const quote=x=>`"${{String(x??"").replaceAll('"','""')}}"`;
 const content=DATA.selection_columns.map(quote).join(",")+"\\n"+selectedIndices.map(i=>{{const row=byIndex.get(i); return DATA.selection_columns.map(c=>quote(row[c])).join(",");}}).join("\\n");
 const link=document.createElement("a"); link.href=URL.createObjectURL(new Blob([content],{{type:"text/csv"}}));
 link.download="molecular_landscape_selected_ids.csv"; link.click(); URL.revokeObjectURL(link.href);
}});
function showMolecule(index) {{
 const row=byIndex.get(index); const neighbors=DATA.neighbors.filter(x=>Number(x.query_structure_index)===index).slice(0,10);
 const local=DATA.cliffs.filter(x=>Number(x.query_structure_index)===index||Number(x.neighbor_structure_index)===index).slice(0,10);
 document.getElementById("molecule-panel").innerHTML=`${{image(row)}}<h3>${{esc(row.compound_id)}}</h3>
 <code>${{esc(row.canonical_smiles)}}</code><p>${{prop?`<b>${{esc(prop)}}:</b> ${{fmt(row[prop])}}<br>`:""}}
 <b>Scaffold:</b> ${{row.scaffold_id}} &nbsp; <b>Cluster:</b> ${{row.cluster_id}}<br>
 <b>MolWt:</b> ${{fmt(row.MolWt)}} &nbsp; <b>LogP:</b> ${{fmt(row.MolLogP??row.LogP)}} &nbsp; <b>TPSA:</b> ${{fmt(row.TPSA)}}<br>
 <b>QED:</b> ${{fmt(row.QED)}} &nbsp; <b>Lipinski violations:</b> ${{fmt(row.Lipinski_Violation_Count)}}</p>
 <h4>Nearest neighbours</h4><table><tr><th>ID</th><th>Tanimoto</th><th>Property Δ</th></tr>${{neighbors.map(n=>`<tr><td>${{esc(n.neighbor_compound_id)}}</td><td>${{fmt(n.tanimoto_similarity)}}</td><td>${{fmt(n.property_difference)}}</td></tr>`).join("")}}</table>
 <h4>Similarity-defined discontinuity partners</h4><table><tr><th>Pair</th><th>Tanimoto</th><th>|Δ|</th></tr>${{local.map(n=>`<tr><td>${{esc(n.query_compound_id)}} / ${{esc(n.neighbor_compound_id)}}</td><td>${{fmt(n.tanimoto_similarity)}}</td><td>${{fmt(n.absolute_property_difference)}}</td></tr>`).join("")}}</table>`;
}}
function highlight(indices) {{ Plotly.restyle("structure-map",{{selectedpoints:[indices.map(i=>DATA.molecules.findIndex(m=>Number(m.structure_index)===Number(i))).filter(i=>i>=0)]}}); location.hash="structure-map"; }}

document.getElementById("scaffold-browser").innerHTML=`<table><tr><th>Representative</th><th>Scaffold</th><th>Size</th><th>Property summary</th><th>Map</th></tr>${{DATA.scaffolds.map(s=>`<tr><td>${{image(byIndex.get(Number(s.representative_structure_index)))}}</td><td><code>${{esc(s.scaffold_smiles||"(acyclic)")}}</code></td><td>${{s.size}}</td><td>min ${{fmt(s.property_min)}}; median ${{fmt(s.property_median)}}; max ${{fmt(s.property_max)}}; active fraction ${{fmt(s.active_fraction)}}</td><td><button onclick='highlight(DATA.molecules.filter(m=>Number(m.scaffold_id)===${{Number(s.scaffold_id)}}).map(m=>m.structure_index))'>Highlight</button></td></tr>`).join("")}}</table>`;
document.getElementById("scaffold-high-low").innerHTML=(DATA.scaffolds_high.length||DATA.scaffolds_low.length)
 ? `<div class="grid"><div><h4>Highest median property</h4><table><tr><th>Scaffold</th><th>n</th><th>Median</th></tr>${{DATA.scaffolds_high.map(s=>`<tr><td>${{s.scaffold_id}}</td><td>${{s.size}}</td><td>${{fmt(s.property_median)}}</td></tr>`).join("")}}</table></div>
 <div><h4>Lowest median property</h4><table><tr><th>Scaffold</th><th>n</th><th>Median</th></tr>${{DATA.scaffolds_low.map(s=>`<tr><td>${{s.scaffold_id}}</td><td>${{s.size}}</td><td>${{fmt(s.property_median)}}</td></tr>`).join("")}}</table></div></div>`
 : `<p class="note">A numeric property is required for scaffold median comparisons.</p>`;
document.getElementById("cluster-browser").innerHTML=`<table><tr><th>Representative</th><th>Cluster</th><th>Size</th><th>Median similarity</th><th>Property summary</th><th>Map</th></tr>${{DATA.clusters.map(c=>`<tr><td>${{image(byIndex.get(Number(c.representative_structure_index)))}}</td><td>${{c.cluster_id}}</td><td>${{c.size}}</td><td>${{fmt(c.median_pairwise_similarity)}}</td><td>min ${{fmt(c.property_min)}}; median ${{fmt(c.property_median)}}; max ${{fmt(c.property_max)}}; active fraction ${{fmt(c.active_fraction)}}</td><td><button onclick='highlight(DATA.molecules.filter(m=>Number(m.cluster_id)===${{Number(c.cluster_id)}}).map(m=>m.structure_index))'>Highlight</button></td></tr>`).join("")}}</table>`;
document.getElementById("cliff-explanation").textContent=DATA.property_profile&&DATA.property_profile.is_potency
 ? "For pChEMBL-like values, a property difference of 1 is approximately a ten-fold potency difference."
 : "These pairs identify structurally similar molecules with unusually different property values; potency language is intentionally avoided.";
document.getElementById("cliff-browser").innerHTML=DATA.cliffs.length?DATA.cliffs.map(c=>`<div class="card"><div class="cliff"><div>${{image(byIndex.get(Number(c.query_structure_index)))}}<b>${{esc(c.query_compound_id)}}</b></div><div>${{image(byIndex.get(Number(c.neighbor_structure_index)))}}<b>${{esc(c.neighbor_compound_id)}}</b></div></div><p>Tanimoto ${{fmt(c.tanimoto_similarity)}}; |Δ| ${{fmt(c.absolute_property_difference)}}; ${{c.same_scaffold?"same":"different"}} scaffold</p><button onclick="highlight([${{Number(c.query_structure_index)}},${{Number(c.neighbor_structure_index)}}])">Highlight pair</button></div>`).join("")
 : "<p>No pairs met the configured similarity-defined discontinuity thresholds.</p>";
const recommendations=DATA.model_readiness?DATA.model_readiness.recommendations:[
 "Use scaffold-aware validation when estimating model generalisation.",
 "Review descriptor outliers, duplicate structures, censored values, and class balance before modelling.",
 "Treat property-aware map geometry as supervised and interpret it only with sensitivity diagnostics."
];
document.getElementById("model-readiness").innerHTML=DATA.model_readiness
 ? `<div class="grid"><div class="metric"><span>Labelled molecules</span><b>${{DATA.model_readiness.n_labelled??"NA"}}</b></div>
 <div class="metric"><span>Singleton scaffold fraction</span><b>${{fmt(DATA.model_readiness.singleton_scaffold_fraction)}}</b></div>
 <div class="metric"><span>Local discontinuities</span><b>${{DATA.model_readiness.local_discontinuity_count}}</b></div></div>
 <ul>${{recommendations.map(x=>`<li>${{esc(x)}}</li>`).join("")}}</ul>`
 : `<p class="note">Run with <code>--advanced-eda</code> or <code>--model-readiness</code> for expanded diagnostics.</p>`;
document.getElementById("reproducibility").innerHTML=`<p><b>Map default:</b> structure-only ${{esc(DATA.map_method)}}.
 <b>Property-aware warning:</b> supervised geometry is not independent evidence of structure-property organisation.</p>`;
showMolecule(Number(DATA.molecules[0].structure_index));
</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")
