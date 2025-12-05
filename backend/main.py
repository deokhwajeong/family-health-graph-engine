from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Household Health Graph API",
    version="0.1.0",
)

# CORS (초기엔 전부 허용, 나중에 도메인 잠그기)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    # 바로 /docs 로 보내기
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health")
def health():
    return {"ok": True}
