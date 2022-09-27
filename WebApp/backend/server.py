from distutils.log import debug
from email.mime import image
from urllib import response
import boto3
import psycopg2

from typing import List
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(debug=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageModel(BaseModel):
    id: int
    image_title: str
    image_url: str

@app.get("/status")
async def status():
    return {"status": "Chardi Kala"}

@app.get("/images", response_model=List[ImageModel])
async def get_images():
    db = psycopg2.connect(
        host="localhost",
        database="mle8_development",
        user="postgres",
        password="postgres"
    )
    cursor = db.cursor()
    cursor.execute("SELECT * FROM images order by id desc")
    images = cursor.fetchall()
    formatted_images = []
    for image in images:
        formatted_images.append(ImageModel(id=image[0], image_title=image[1], image_url=image[2]))
    cursor.close()
    db.close()
    return formatted_images

@app.post("/images", status_code=201)
async def add_photo(file: UploadFile):
    # Upload the file to AWS S3
    s3 = boto3.client('s3')
    bucket = s3.create_bucket(Bucket='mle8-images')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
