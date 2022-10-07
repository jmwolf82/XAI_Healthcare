from distutils.log import debug
from email.mime import image
from urllib import response
# import boto3
import psycopg2

import logging
import boto3
from botocore.exceptions import ClientError
import os

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

S3_BUCKET_NAME = 'mle8-images'

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
        formatted_images.append(
            ImageModel(id=image[0], image_title=image[1], image_url=image[2])
        )
    cursor.close()
    db.close()
    return formatted_images

@app.post("/images", status_code=201)
async def add_photo(file: UploadFile):
    # Upload the file to AWS S3
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(S3_BUCKET_NAME)
    bucket.upload_fileobj(file.file, file.filename, ExtraArgs={'ACL': 'public-read'})
    uploaded_file_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{file.filename}"

    # Store the file to the database
    db = psycopg2.connect(host="localhost", database="mle8_development", user="postgres", password="postgres")
    cursor = db.cursor()
    cursor.execute("INSERT INTO images (image_title, image_url) VALUES (%s, %s)", (file.filename, uploaded_file_url))
    db.commit()
    cursor.close()
    db.close()
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
