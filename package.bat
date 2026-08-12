@echo off
rem ============================================================
rem  一键打包 EchoGuide 主要代码，供网页版 AI 分析
rem  产出: dist\echoguide-code-*.zip（内含 _AI_SUMMARY.md 摘要）
rem  排除: node_modules / .git / .venv / 密钥 / 截图 / PDF 等
rem ============================================================
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py scripts\package_code.py
) else (
    python scripts\package_code.py
)

if errorlevel 1 (
    echo.
    echo 打包失败，请确认已安装 Python 并加入 PATH。
) else (
    echo.
    echo 打包完成，压缩包在 dist 目录，可直接上传给网页版 AI。
)
pause
