import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse

from application.components import predict, read_imagefile
from application.schema import Symptom
from application.components.prediction import symptom_check

app_desc = """<h2>Use this 🦠  API by uploading any image with `predict/image`</h2>
<h4> 🦠 Brain Tumor Prediction API - it is just for research learning</h4>
<br>by Olusegun Odewole - AIOT lab"""

app = FastAPI(title=' 🦠 Brain Tumor Prediction API', description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)

    return prediction




if __name__ == "__main__":
    uvicorn.run(app, debug=True)
