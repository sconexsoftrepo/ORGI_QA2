from fastapi import FastAPI
from fastapi.responses import JSONResponse
import sys
import main

app = FastAPI(
    title="Pipeline API",
    version="1.0"
)

# Fixed internally — not exposed as an API parameter.
MAX_BATCH_SIZE = 50


@app.post("/run")
def run_pipeline():
    """
    Trigger a pipeline run.

    Batch size is capped at MAX_BATCH_SIZE (50) internally. main.py
    recomputes the actual batch size every loop iteration as
    min(MAX_BATCH_SIZE, unprocessed_stores), so it naturally handles
    1 store, 10 stores, 51 stores, etc. without needing input() anymore.
    """

    POD_ID = "pod-1"

    sys.argv = [
        "main.py",
        POD_ID,
        str(MAX_BATCH_SIZE),
    ]

    try:

        main.main()

        return JSONResponse(
            {
                "status": "completed",
                "pod": POD_ID,
                "max_batch_size": MAX_BATCH_SIZE,
            }
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )