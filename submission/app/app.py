"""
Logo tampering check: Streamlit app for the final model (e14_consistent, ONNX).

Load a reference and a photo -> the app aligns the photo with the reference
(EXP-001), runs the model, and shows the decision, the probability, the tamper
type probabilities and, optionally, the heatmap of where the model sees a difference.

Run (from the submission folder):
    streamlit run app/app.py
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import pipeline as P

HERE = Path(__file__).resolve().parent
SAMPLES = HERE.parent / "samples"                               # the 50 photos of the assignment


def read_upload(upload):
    data = np.frombuffer(upload.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def show_image(bgr, caption, max_width=420):
    """Display a BGR image without stretching to the full page width."""
    st.image(rgb(bgr), caption=caption, width=max_width)


st.set_page_config(page_title="Logo tampering check", layout="wide")
st.title("Logo tampering check")

# ---- inputs -----------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("1. Reference")
    ref_choice = st.radio("Reference logo", ["All 3 built-in references (recommended)", "Built-in ref_1", "Upload my own"],
                          horizontal=False)
    if ref_choice == "Upload my own":
        up = st.file_uploader("Reference image", type=["jpg", "jpeg", "png"])
        references = [read_upload(up)] if up else []
    elif ref_choice == "Built-in ref_1":
        references = [cv2.imread(str(HERE / "references" / "ref_1.jpg"))]
    else:
        references = [cv2.imread(str(HERE / "references" / f"ref_{i}.jpg")) for i in (1, 2, 3)]
    if references:
        show_image(P.logo_template(references[0]), "Reference", max_width=360)

with right:
    st.subheader("2. Photo to check")
    source = st.radio("Photo", ["Upload a photo"] + (["Pick one of the 50 test photos"] if SAMPLES.exists() else []))
    photo = None
    if source == "Upload a photo":
        up = st.file_uploader("Photo of the label", type=["jpg", "jpeg", "png"], key="photo")
        photo = read_upload(up) if up else None
    else:
        name = st.selectbox("Test photo", sorted(p.name for p in SAMPLES.glob("*.jpg")))
        photo = cv2.imread(str(SAMPLES / name))
    if photo is not None:
        show_image(photo, "Photo", max_width=360)

show_heat = st.checkbox("Show heatmap", value=True)

# ---- run --------------------------------------------------------------------
if references and photo is not None and st.button("Check this photo", type="primary"):
    with st.spinner("Checking…"):
        a = P.align(references[0], photo)
    if not a["ok"]:
        st.info("This photo could not be checked.")
        st.stop()

    templates = [P.logo_template(r) for r in references]
    r = P.predict(templates, a["crop"])

    st.subheader("3. Result")
    colour = {"TAMPERED": "red", "GENUINE": "green", "DON'T KNOW": "orange"}[r["decision"]]
    st.markdown(f"### :{colour}[{r['decision']}]  —  p(tampered) = {r['p']:.3f}")

    photo_box = photo.copy()
    cv2.polylines(photo_box, [a["outline"].astype(np.int32)], True, (0, 220, 255), 4)
    cols = st.columns(3 if show_heat else 2)
    with cols[0]:
        show_image(photo_box, "Logo location")
    with cols[1]:
        show_image(P.overlay_outline(a["template"], a["crop"]), "Aligned logo")
    if show_heat:
        with cols[2]:
            show_image(P.heatmap_overlay(a["crop"], r["heatmap"]), "Heatmap")
