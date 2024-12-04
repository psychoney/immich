import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "face_service:app",
        host="0.0.0.0",
        port=3011,
        reload=True
    ) 