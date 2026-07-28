# Images

`dashboard_preview.png` is the screenshot embedded at the top of the project README.
It has to be captured by hand — no script can produce it.

To (re)generate it:

1. `streamlit run app.py` from the project root.
2. Open the **Executive Summary** page and leave the sidebar at its default thresholds.
3. Widen the browser window so the KPI row sits on one line (~1600px works well).
4. Screenshot the page and save it here as `dashboard_preview.png`.

The app pins a light theme in `.streamlit/config.toml`, so the screenshot will match the
chart palette regardless of your OS appearance setting.
