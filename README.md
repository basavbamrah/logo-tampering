# Logo tampering detection

Detects whether the NORVIA logo on a product photo has been tampered with. The model aligns the photo with a clean reference, then scores it as **TAMPERED**, **GENUINE**, or **DON'T KNOW**.

Everything you need to run, check, and read the work lives in **`submission/`**. That folder is self-contained: model (ONNX), app, sample photos, write-up, and experiment notes.

## Start here

Read **[submission/README.md](submission/README.md)** for install steps, how to check the model, results on the 50 given photos, and links to the write-up and experiments.

## Run the app

From the repository root:

```
cd submission
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Then open the address Streamlit prints (normally http://localhost:8501). Choose a reference, pick or upload a photo, and press **Check this photo**.

PyTorch is not required to run the app. See [submission/README.md](submission/README.md) for the accuracy check, re-running the 50 photos, and ONNX export.
