from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os

app = FastAPI(
    title="ChatPression",
    description="AI 첫인상 분석 서비스",
    version="1.0.0"
)

# 프로젝트 루트 디렉토리 경로 설정
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 정적 파일 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory=os.path.join(ROOT_DIR, "frontend/static")), name="static")

# 템플릿 설정 (HTML 파일)
templates = Jinja2Templates(directory=os.path.join(ROOT_DIR, "frontend/templates"))


@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# 사용자 정보 입력 페이지
@app.get("/user-info", response_class=HTMLResponse)
async def read_user_info(request: Request):
    return templates.TemplateResponse(
        "user-info.html",
        {"request": request}
    )


@app.get("/conversation", response_class=HTMLResponse)
async def read_conversation(request: Request):
    return templates.TemplateResponse(
        "conversation.html",
        {"request": request}
    )


@app.get("/result", response_class=HTMLResponse)
async def read_result(request: Request):
    return templates.TemplateResponse(
        "result.html",
        {"request": request}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

