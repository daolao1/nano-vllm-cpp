#!/usr/bin/env bash
# 启动本地 HTTP 服务器，用于查看 docs/ 和 question/ 下的 HTML 文件
# 用法: ./serve_docs.sh [端口号]  (默认 8080)

PORT="${1:-8080}"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================="
echo "  文档服务器已启动"
echo "  http://localhost:${PORT}"
echo ""
echo "  参数文档:  http://localhost:${PORT}/docs/config_params.html"
echo "  面试题:    http://localhost:${PORT}/question/config_interview.html"
echo ""
echo "  Ctrl+C 停止"
echo "========================================="

cd "$DIR" && python3 -m http.server "$PORT"
