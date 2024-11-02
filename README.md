# Figi

This is the backend for [Figi](https://github.com/elijahfhopp/figi), a local-first face search engine. It is build with Python + FastAPI + OpenCV + Strawberry + pgvector + Peewee.

## DB Config

Currently the db connection config is all hard-coded. If you want to deploy it in somewhere other than a `localhost` Postgres instance, it will take some code changes.

## Setup

Postgres + pgvector:
1. Should be as simple as `docker compose up -d`.

Figi pre-flight:
1. Get Python and Poetry installed.
2. Download the weight files for [SFace](https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) and [YuNet](https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx).
3. [Configure the environment variables.](#environment-variables)

Figi:
1. `poetry install --no-root` and `poetry shell`.
2. Start the server with indexing: `FIGI_INDEX=1 uvicorn main:app`
3. Smoke test with `wget http://localhost:8000/image/0 && file 0.jpg`?
4. [Set up the frontend?](https://github.com/elijahfhopp/figi-frontend)

## Environment variables

The environment variables are set in `.env` or on the command line. Here is a complete list of config keys (defaults):

```
FIGI_INDEX=0
FIGI_IMAGES_PATH=images/
FIGI_MODEL_PATH=.
FIGI_MAX_VECTOR_SEARCH_LIMIT=100
```

Image indexing only happens at start up if `FIGI_INDEX` is `True`