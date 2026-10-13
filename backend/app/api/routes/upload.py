from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.data_service import save_and_parse

router = APIRouter()


@router.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=422, detail="仅支持 .xlsx / .xls 格式")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件过大，最大支持 50MB")
    try:
        result = save_and_parse(content, file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"文件解析失败: {str(e)[:200]}")
