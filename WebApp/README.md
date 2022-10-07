# Front End and Backend for Project Healthcare

## Process

1. User Uplods image through Web Interface
2. The image is stored in AWS S3
3. The image URL is stored in Postgres
4. The image is chopped into 64x64px tiles and numbered. The tile co-ordinates are stored in Postgres. For now the remainder edges are ignored
5. The images are tested against a trained model returning mitotic or non-mitotic
6. The returned images are assembled together (the assembled images may have some annotations)
7. Any text annotations are assembled together are displayed along with the image.
8. The process will likely be asynchronous.

## Backend 
- FastAPI, 
- PostgreSQL
- AWS S3

## Frontend 
- React JS 
- Chakra UI

# Training Model
- EC2