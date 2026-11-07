"""Freeze new event-level training/test references BEFORE any new predictions."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "positive_expansion_20261001"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, array):
    ok, data = cv2.imencode(".png", array)
    if not ok:
        raise ValueError("Image encoding failed")
    path.write_bytes(data.tobytes())


def polygon(shape, points):
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.asarray(points, np.int32)], 1)
    return result


def clean_windows(allowed, artefacts, size=32):
    valid = np.zeros_like(allowed, np.uint8)
    for y in range(0, allowed.shape[0]-size+1, size):
        for x in range(0, allowed.shape[1]-size+1, size):
            sl = np.s_[y:y+size, x:x+size]
            if allowed[sl].all() and not artefacts[sl].any():
                valid[sl] = 1
    return valid


def prepare():
    manifest = OUT / "positive_cases.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        for name, digest in existing["frozen_files"].items():
            if sha(OUT / name) != digest:
                raise ValueError(f"Frozen positive asset changed: {name}")
        return existing
    cases = OUT / "cases"
    cases.mkdir(parents=True, exist_ok=True)
    import fitz
    # The report stores optical imagery as image strips, independent of annotation
    # vectors. Keep original strip pixels; skip the overlapping final strip row.
    doc = fitz.open(OUT / "sources/collolar_report.pdf")
    rows = []
    for item in doc[0].get_images(full=True):
        if item[2] != 1812:
            continue
        pix = fitz.Pixmap(doc, item[0])
        rect = doc[0].get_image_rects(item[0])[0]
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        rows.append((rect.y0, rgb[:-1]))
    rows.sort(key=lambda r: r[0])
    optical = np.concatenate([r[1] for r in rows])[:, :, ::-1].copy()
    seam_y = np.cumsum([len(r[1]) for r in rows])[:-1]
    # Conservative interior of the failed mass indicated by UNOSAT as Landslide A.
    # These are author-event-informed weak interiors, NOT a complete expert mask.
    points = [[900,160],[1090,130],[1220,260],[1260,450],[1170,650],
              [1310,825],[1240,1000],[960,1020],[820,890],[820,710],
              [870,550],[805,370]]
    interior = polygon(optical.shape[:2], points)
    interior = cv2.erode(interior, np.ones((25,25), np.uint8))
    seams = np.zeros_like(interior)
    for y in seam_y:
        seams[max(0,y-3):y+4] = 1
    chosen = clean_windows(interior, seams)
    patches, provenance = [], []
    for y in range(0, len(optical)-31, 32):
        for x in range(0, optical.shape[1]-31, 32):
            if chosen[y:y+32,x:x+32].all():
                bgr = optical[y:y+32,x:x+32]
                patches.append(np.concatenate([bgr,bgr[:,:,2:3]],2).transpose(2,0,1))
                provenance.append({"event_id":"collolar_201102", "box_xyxy":[x,y,x+32,y+32],
                                   "label":"weak_positive_interior", "independent_event_count":1})
    if len(patches) < 100:
        raise ValueError("Insufficient conservative positive training interiors")
    np.save(OUT / "collolar_positive_x.npy", np.stack(patches).astype(np.float32))
    np.save(OUT / "collolar_positive_y.npy", np.ones((len(patches),32,32),np.float32))
    save(cases / "collolar_optical_no_annotations.png", optical)
    save(cases / "collolar_training_interiors.png", chosen*255)
    review = optical.copy()
    review[chosen>0] = (.55*review[chosen>0]+.45*np.array([40,200,40])).astype(np.uint8)
    save(cases / "collolar_training_review.png", review)

    # New held-out mine: only collapsed-source zones are positive. The remaining
    # red runout footprint may contain affected material or water; leave it unknown.
    # Digitization follows the published BLUE dashed boundary, not predictions.
    fig = cv2.imdecode(np.fromfile(OUT / "sources/hpakant_figure7.jpg",np.uint8),1)
    crop = [1535,698,2558,1385]
    image = fig[crop[1]:crop[3],crop[0]:crop[2]].copy()
    west = [[126,170],[142,146],[191,131],[231,117],[271,102],[300,77],
            [302,128],[284,183],[270,229],[271,285],[281,325],[274,360],
            [239,361],[199,352],[167,346],[126,330],[132,275],[121,258],
            [144,221],[137,192]]
    north = [[301,77],[340,76],[374,59],[422,69],[470,81],[518,97],
             [562,73],[611,72],[656,78],[697,76],[714,51],[735,25],
             [782,32],[832,44],[827,78],[816,103],[779,107],[742,113],
             [710,132],[674,146],[648,159],[615,144],[590,135],[552,140],
             [516,148],[478,160],[455,180],[429,194],[394,182],[353,171],
             [329,143],[313,108]]
    target = polygon(image.shape[:2], west) | polygon(image.shape[:2], north)
    b,g,r = [c.astype(np.int16) for c in cv2.split(image)]
    red = ((r>110)&(r-g>50)&(r-b>50)).astype(np.uint8)
    # The common red/blue northern line is intentionally dashed. Digitize its
    # vertices instead of closing color fragments across arbitrary image objects.
    runout_points = [[126,170],[142,146],[231,117],[271,102],[300,77],
        [340,76],[374,59],[470,81],[518,97],[562,73],[656,78],[697,76],
        [714,51],[735,25],[782,32],[832,44],[864,67],[888,96],[904,127],
        [909,174],[902,215],[885,242],[831,270],[754,273],[680,285],
        [608,279],[578,323],[567,360],[562,398],[498,433],[480,453],
        [466,494],[468,518],[491,548],[480,560],[389,586],[321,616],
        [254,637],[211,633],[147,611],[114,593],[98,568],[82,538],
        [81,496],[116,466],[124,426],[137,388],[122,350],[117,309],
        [124,273],[121,258],[144,221],[137,192]]
    runout = polygon(image.shape[:2],runout_points)
    if not .3 < runout.mean() < .8:
        raise ValueError("Runout boundary extraction failed")
    target &= runout
    # Annotation exclusion: all saturated colored lines/markers, text corners,
    # manually digitized polygon uncertainty (+/- 10 pixels), and partial tiles.
    # Do not mistake the blue cast of actual Planet imagery for drawn blue lines.
    # The polygon uncertainty band below excludes the blue boundary itself.
    green = ((g>180)&(g-r>80)&(g-b>80)).astype(np.uint8)
    artefacts = cv2.dilate(red|green,np.ones((7,7),np.uint8))
    edge = cv2.morphologyEx(target,cv2.MORPH_GRADIENT,np.ones((21,21),np.uint8))
    artefacts |= edge
    artefacts[:100,:100] = 1
    artefacts[600:,700:] = 1
    allowed = target | (1-cv2.dilate(runout,np.ones((21,21),np.uint8)))
    valid = clean_windows(allowed,artefacts)
    print(f"Hpakant clean foreground={int((target&valid).sum())}, background={int(((1-target)&valid).sum())}")
    if int((target&valid).sum()) < 4096 or int(((1-target)&valid).sum()) < 4096:
        raise ValueError("Too few clean positive/background windows")
    save(cases / "hpakant_post_20200803.png",image)
    save(cases / "hpakant_collapsed_reference.png",target*255)
    save(cases / "hpakant_valid.png",valid*255)
    save(cases / "hpakant_runout_unknown.png",((runout>0)&(target==0)).astype(np.uint8)*255)
    review = image.copy()
    review[valid==0] = (.3*review[valid==0]).astype(np.uint8)
    review[(target&valid)>0] = (.55*review[(target&valid)>0]+.45*np.array([40,210,40])).astype(np.uint8)
    save(cases / "hpakant_reference_review.png",review)
    paths = ["collolar_positive_x.npy","collolar_positive_y.npy",
             "sources/collolar_report.pdf","sources/hpakant_figure7.jpg"]
    paths += [str(p.relative_to(OUT)).replace("\\","/") for p in cases.glob("*.png")]
    result = {"version":1,"frozen_before_training_and_predictions":True,
              "new_independent_mines":2,"new_training_mines":1,"new_test_mines":1,
              "original_las_cruces_and_bingham_held_out":True,
              "full_expert_audited_test_sets":0,
              "collolar":{"event":"2011-02-06/10","acquisition":"2011-02-17",
                  "source":"https://unosat.org/products/2117","report_version":"08 March 2011 v2",
                  "label_quality":"manual conservative weak interiors informed by official event assessment",
                  "role":"training only; not used for checkpoint/threshold selection",
                  "raw_shape":list(optical.shape),"positive_patch_n":len(patches),
                  "weak_label_weight":.25,"interior_polygon_xy":points,
                  "seam_rows_excluded":seam_y.tolist(),"patches":provenance,
                  "imagery_credit":"WorldView-2 / DigitalGlobe 2011 / USGS HDDS; UNITAR UNOSAT assessment",
                  "license":"public assessment; third-party imagery copyright retained, no open dataset redistribution license asserted",
                  "ground_validation":"UNOSAT explicitly states not independently validated on the ground"},
              "hpakant":{"event":"2020-07-02","acquisition":"2020-08-03",
                  "source":"https://doi.org/10.1016/j.isprsjprs.2021.05.015",
                  "source_figure":"Figure 7D, Lin et al. 2021",
                  "source_license":"Article CC BY 4.0; underlying Planet imagery rights retained",
                  "role":"new held-out test, no training/selection/tuning",
                  "target_semantics":"collapsed source zones (blue dashed); other runout footprint unknown",
                  "label_quality":"manually digitized published figure, no independent expert audit",
                  "reference_polygons_xy":[west,north],"runout_polygon_xy":runout_points,"crop_xyxy":crop,
                  "image":"cases/hpakant_post_20200803.png",
                  "reference":"cases/hpakant_collapsed_reference.png","valid":"cases/hpakant_valid.png",
                  "scored_positive_pixels":int((target&valid).sum()),
                  "scored_background_pixels":int(((1-target)&valid).sum())},
              "frozen_files":{p:sha(OUT/p) for p in paths}}
    manifest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Frozen: {len(patches)} weak-positive training patches / 1 mine; Hpakant test foreground={(target&valid).sum()}, background={((1-target)&valid).sum()}")
    return result


if __name__ == "__main__":
    prepare()
