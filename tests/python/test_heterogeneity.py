"""Independent cell arithmetic and outcome-scale interaction regression checks."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "engines/python"), str(ROOT / "sidecar")]
from capy_py.contracts import run_method
from capy_py.heterogeneity import holm, wald_test
from capy_sidecar import examples, registry, store, compare


def spec(outcome="y", confounders=None):
    return {"schema":"capy.spec","version":1,"id":"heterogeneity_test","design":"rct","estimand":"GATE",
            "roles":{"treatment":"a","outcome":outcome,"confounders":confounders or []}}


OPTIONS = {"subgroups":["g"],"subgroups_prespecified":True}


def cells(family="binary"):
    rows=[]
    for g in (0,1):
        for a in (0,1):
            if family == "binary":
                events={(0,0):20,(0,1):40,(1,0):40,(1,1):50}[g,a]
                values=[1]*events+[0]*(100-events)
            elif family == "continuous":
                values=np.linspace(-2,2,100)+g+a*(1+g)
            elif family == "count":
                values=np.tile(np.array([0,1,2,3])+g+a*(1+g),25)
            else:
                proportions={(0,0):[30,30,40],(0,1):[20,40,40],(1,0):[20,40,40],(1,1):[40,40,20]}[g,a]
                values=sum(([label]*n for label,n in zip(["low","middle","high"],proportions)),[])
            rows.extend({"g":g,"a":a,"y":y} for y in values)
    return pd.DataFrame(rows)


def run(family, df=None, options=None, study=None):
    result=run_method("heterogeneity."+family,study or spec(),cells(family) if df is None else df,options={**OPTIONS,**(options or {})})
    assert result["status"]=="ok",result.get("error")
    import jsonschema
    jsonschema.validate(result,json.loads((ROOT/"schemas/capy.result.v1.json").read_text()))
    json.dumps(result,allow_nan=False)
    return result


def diagnostic(result,which="outcome_heterogeneity"):
    return next(d["values"] for d in result["diagnostics"] if d["id"]==which)


def rows(result,ident="heterogeneity_subgroups"):
    return next(a["data"] for a in result["artifacts"] if a["id"]==ident)


def test_binary_risk_difference_matches_independent_saturated_cell_wald():
    result=run("binary")
    assert [r["estimate"] for r in rows(result)]==pytest.approx([.2,.1],abs=1e-10)
    variance=sum(p*(1-p)/100 for p in [.2,.4,.4,.5])
    assert diagnostic(result)["statistic"]==pytest.approx(.1**2/variance,rel=2e-5)
    assert diagnostic(result)["p_value"]==pytest.approx(chi2.sf(.1**2/variance,1),rel=2e-5)
    assert diagnostic(result,"link_heterogeneity")["p_value"] != pytest.approx(diagnostic(result)["p_value"],rel=.01)


@pytest.mark.parametrize("scale,expected",[("risk_ratio",[2.,1.25]),("odds_ratio",[8/3,1.5])])
def test_binary_ratios_and_log_scale_intervals(scale,expected):
    result=run("binary",options={"effect_scale":scale})
    assert [r["estimate"] for r in rows(result)]==pytest.approx(expected,rel=1e-7)
    assert result["heterogeneity_target"]["null"]==1
    assert all(r["ci_low"]>0 for r in rows(result))
    assert rows(result,"heterogeneity_pairwise")[0]["difference_on_test_scale"]==pytest.approx(np.log(expected[0]/expected[1]),rel=1e-7)


def test_continuous_hc3_matches_independent_cell_variances():
    df=cells("continuous")
    result=run("continuous",df)
    assert [r["estimate"] for r in rows(result)]==pytest.approx([1,2])
    var=sum(float(np.square(c.y-c.y.mean()).sum())/(len(c)-1)**2 for _,c in df.groupby(["g","a"]))
    assert diagnostic(result)["statistic"]==pytest.approx(1/var,rel=2e-5)
    assert result["estimate"]==pytest.approx(1.5)


def test_count_mean_and_ratio_match_cell_means():
    df=cells("count")
    means=df.groupby(["g","a"]).y.mean()
    difference=run("count",df)
    ratio=run("count",df,options={"effect_scale":"mean_ratio"})
    assert [r["estimate"] for r in rows(difference)]==pytest.approx([means[g,1]-means[g,0] for g in (0,1)])
    assert [r["estimate"] for r in rows(ratio)]==pytest.approx([means[g,1]/means[g,0] for g in (0,1)])


def test_count_exposure_predicts_rates_and_refuses_nonpositive_exposure():
    df=cells("count")
    df["exposure"]=np.where(df.a==1,2.,1.)
    result=run("count",df,options={"exposure":"exposure"})
    assert [r["estimate"] for r in rows(result)]==pytest.approx([2.5/2-1.5,4.5/2-2.5])
    assert "rate difference" in result["estimand_label"]
    ratio=run("count",df,options={"exposure":"exposure","effect_scale":"mean_ratio"})
    assert "rate ratio (dimensionless)" in ratio["estimand_label"]
    from capy_sidecar import compare, reports
    assert compare._effect_axis([ratio]) == "Rate ratio (dimensionless; null = 1)"
    assert "ratio is dimensionless" in reports.honest_paragraph([ratio])
    rescaled=run("count",df.assign(exposure=df.exposure*10),options={"exposure":"exposure","effect_scale":"mean_ratio"})
    assert rescaled["estimate"] == pytest.approx(ratio["estimate"],rel=1e-7)
    assert diagnostic(rescaled)["statistic"] == pytest.approx(diagnostic(ratio)["statistic"],rel=1e-6)
    df.loc[0,"exposure"]=0
    failure=run_method("heterogeneity.count",spec(),df,options={**OPTIONS,"exposure":"exposure"})
    assert failure["status"]=="failed" and "positive" in failure["error"]["message"]


def test_multinomial_probabilities_sum_zero_and_joint_covariance_oracle():
    df=cells("categorical")
    result=run("categorical",df)
    subgroup=rows(result)
    assert result["estimate"] is None
    assert len(subgroup)==6
    assert all(sum(r["estimate"] for r in subgroup if r["subgroup"]==str(g))==pytest.approx(0,abs=1e-10) for g in (0,1))
    categories=sorted(df.y.unique())
    probs={(g,a):df[(df.g==g)&(df.a==a)].y.value_counts().reindex(categories).to_numpy()/100 for g in (0,1) for a in (0,1)}
    d=(probs[1,1]-probs[1,0])-(probs[0,1]-probs[0,0])
    v=sum((np.diag(p)-np.outer(p,p))/100 for p in probs.values())
    oracle=float(d[:-1]@np.linalg.solve(v[:-1,:-1],d[:-1]))
    assert diagnostic(result)["statistic"]==pytest.approx(oracle,rel=2e-5)
    assert diagnostic(result)["df"]==2
    assert len(rows(result,"heterogeneity_pairwise"))==3


def test_multinomial_is_invariant_to_outcome_category_labels():
    df=cells("categorical")
    original=run("categorical",df)
    df.y=df.y.map({"low":"z","middle":"a","high":"m"})
    changed=run("categorical",df)
    assert diagnostic(changed)["statistic"]==pytest.approx(diagnostic(original)["statistic"],rel=2e-5)


@pytest.mark.parametrize("family",["continuous","binary","count","categorical"])
def test_all_families_synthetic_adjustment_continuous_modifier_and_determinism(family):
    df, assigned,_,_=examples._heterogeneous_outcomes()
    assigned["outcome"]={"continuous":"continuous_outcome","binary":"binary_outcome","count":"count_outcome","categorical":"category_outcome"}[family]
    s=spec();s["roles"]=assigned
    opts={"subgroups":["baseline_score"],"modifier_type":"continuous","modifier_low":-1.,"modifier_high":1.}
    first=run(family,df,opts,s);second=run(family,df,opts,s)
    assert diagnostic(first)==diagnostic(second)
    assert diagnostic(first)["levels"]==["baseline_score = -1","baseline_score = 1"]
    assert first["heterogeneity_target"]["inference_target"]=="conditional on empirical baseline profiles"


@pytest.mark.parametrize("family",["continuous","binary","count","categorical"])
def test_cluster_sandwich_and_too_few_clusters(family):
    df=cells(family); df["cluster"]=np.random.default_rng(826).permutation(np.arange(len(df))%40)
    s=spec();s["roles"]["cluster"]="cluster"
    result=run(family,df,study=s)
    assert "Cluster" in result["inference"]
    df["cluster"]=1
    failure=run_method("heterogeneity."+family,s,df,options=OPTIONS)
    assert failure["status"]=="failed" and "twenty" in failure["error"]["message"]


@pytest.mark.parametrize("change,fragment",[
    ({"subgroups":[]},"exactly one"),({"subgroups":["g","g"]},"exactly one"),
    ({"subgroups_prespecified":False},"Confirm"),({"subgroups_prespecified":"true"},"Confirm"),
    ({"subgroups":["y"]},"distinct"),({"effect_scale":"hazard_ratio"},"effect scale"),
    ({"ci_level":2},"Confidence"),({"modifier_type":"continuous","modifier_low":-1,"modifier_high":2},"observed range"),
])
def test_invalid_options_fail_clearly(change,fragment):
    result=run_method("heterogeneity.binary",spec(),cells(),options={**OPTIONS,**change})
    assert result["status"]=="failed"
    assert fragment in result["error"]["message"]


@pytest.mark.parametrize("family",["continuous","binary","count","categorical"])
def test_dropped_rows_and_forbidden_modifier(family):
    df=cells(family);df.loc[0,"g"]=np.nan
    result=run(family,df)
    assert result["n"]==len(df)-1 and result["sample_flow"][-1]["n"]==len(df)-1
    s=spec();s["roles"]["forbidden"]=["g"]
    failure=run_method("heterogeneity."+family,s,df,options=OPTIONS)
    assert failure["status"]=="failed" and "forbidden" in failure["error"]["message"]


def test_separation_and_invalid_outcome_are_refused():
    df=cells();df.y=df.a
    result=run_method("heterogeneity.binary",spec(),df,options=OPTIONS)
    assert result["status"]=="failed" and "non-events" in result["error"]["message"]
    df=cells("count");df.loc[0,"y"]=-1
    result=run_method("heterogeneity.count",spec(),df,options=OPTIONS)
    assert result["status"]=="failed" and "nonnegative integers" in result["error"]["message"]


def test_holm_known_order_and_singular_covariance_abstention():
    assert holm([.04,.01,.03])==pytest.approx([.06,.03,.06])
    assert wald_test([1,1],[[1,1],[1,1]])["test_status"]=="not_estimable"
    assert wald_test([1,1],[[1,1],[1,1]])["p_value"] is None


def test_two_group_p_values_are_direct_difference_test():
    result=run("binary")
    pair=rows(result,"heterogeneity_pairwise")[0]
    assert pair["p_value_holm"]==pytest.approx(diagnostic(result)["p_value"],rel=1e-6)
    assert pair["interval_scope"].startswith("pointwise")


def test_registered_cards_example_and_summary_preserve_targets():
    cards={c["id"]:c for c in registry.methods()}
    for family in ("continuous","binary","count","categorical"):
        assert cards["heterogeneity."+family]["python_available"]
    result=run("binary",options={"effect_scale":"risk_ratio"})
    assert store.run_summary(result)["heterogeneity_target"]==result["heterogeneity_target"]
    df,r,notes=examples.dataset("heterogeneous_outcomes")
    assert len(df)==1800 and r["outcome"]=="continuous_outcome"
    assert notes and examples.BY_ID["heterogeneous_outcomes"]["source"]=="simulated"
    recommended=registry.recommend(spec())
    assert not any(c["id"].startswith("heterogeneity.") for c in recommended["recommended"])
    assert len([c for c in recommended["reasonable"] if c["id"].startswith("heterogeneity.")])==4


def test_outcome_scale_comparisons_do_not_pool_unlike_effects():
    additive=run("binary");ratio=run("binary",options={"effect_scale":"risk_ratio"})
    assert any(not c["ok"] for c in compare._comparability([additive,ratio]))
    assert compare._spread([additive,ratio]) is None
    assert "vconcat" in compare._forest([additive,ratio])


@pytest.mark.parametrize("family",["continuous","binary","count","categorical"])
@pytest.mark.parametrize("fmt",["markdown","html","latex","docx"])
def test_exports_preserve_outcome_target_and_direct_test(family,fmt,tmp_path,monkeypatch):
    import base64,io,zipfile
    from capy_sidecar import reports,settings
    monkeypatch.setattr(store,"SETTINGS",settings.Settings(home=tmp_path/"home",workspace=tmp_path/"workspace"))
    project=store.Store().create("Outcome heterogeneity export")
    study=project.save_spec(spec())
    result=run(family,study=study)
    project.save_run(result)
    report=reports.default_report(project,study["id"],[result["run_id"]])
    exported=reports.render(project,report,fmt=fmt,save=False)
    text=exported["content"]
    if fmt=="docx":
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(text))) as archive:
            text=archive.read("word/document.xml").decode()
    assert "Holm" in text and "empirical baseline" in text
    assert family in text
    assert "pointwise" in text.lower()
    assert "category-probability vector" in text if family=="categorical" else "overall standardized contrast" in text


def test_closed_option_bounds_reject_noninteger_and_extra_fields():
    sys.path.insert(0,str(ROOT/"knowledge"))
    from capy_knowledge.compiler import _control_value
    from capy_knowledge.contracts import KnowledgeError
    contract={"type":["integer"],"minimum":1,"maximum":8}
    _control_value(1,contract)
    for value in (True,1.,0,9,"1"):
        with pytest.raises(KnowledgeError):
            _control_value(value,contract)
