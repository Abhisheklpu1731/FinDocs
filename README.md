
## Run the app

```
uv run streamlit run app.py
```

- **Ask** tab: chat with the selected reports. Each answer shows live pipeline steps, cited sources, and the original PDF page.
- **Add a report** tab: upload any company's annual report (PDF). It is parsed, chunked, embedded and indexed with a live progress bar, then saved under `data/uploads/` so it survives restarts. Remove it any time from the sidebar.
- Scanned (image-only) PDFs have no OCR step yet; pages without a text layer are skipped and reported.
