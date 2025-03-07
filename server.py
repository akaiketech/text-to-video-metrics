from fastapi import FastAPI
from .routers import video_quality, alignment
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Video Analysis API")





app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains (adjust for your specific use case)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

app.include_router(video_quality.router, prefix="/video-quality")
app.include_router(alignment.router, prefix="/text-video-alignment", tags=["Text-Video Alignment"])

@app.get("/")
def read_root():
    return {"message": "Video Evaluation API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
