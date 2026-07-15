"""
wsgi.py - WSGI 入口(Gunicorn 加载) / WSGI entry (Gunicorn)

Gunicorn 通过 `wsgi:app` 加载此文件,从而拿到 Flask 应用实例。 / Gunicorn loads
this file via `wsgi:app` to obtain the Flask app instance.

功能模块 / Modules:
- 导出 Flask `app` 对象 / Export Flask `app` object

输入 / Inputs:
- 无 / None

输出 / Outputs:
- `app`: Flask 实例 / Flask instance

数据流 / Data Flow:
1. Gunicorn 启动 → import wsgi → 触发 server.py 的 app 创建 / Gunicorn imports wsgi → triggers app init
2. server.py 完成 Redis、模型、日志初始化 / server.py completes init

相关文件 / Related Files:
- 调用 / Calls: server(导入 app)
- 被调用 / Called by: Gunicorn(`gunicorn wsgi:app`)、start_backend.sh

使用示例 / Usage Example:
    gunicorn --bind 0.0.0.0:8000 --workers 4 wsgi:app

作者 / Author: 项目组 / Project Team
版本 / Version: 1.0
"""
"""
WSGI entry point for Gunicorn production server.

This module exports the Flask application for use with Gunicorn.
"""
from server import app

if __name__ == "__main__":
    app.run()
