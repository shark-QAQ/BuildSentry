# ui/app.py
"""
前端应用文件，包含 Gradio 实例和组件。

后端接口除登录外均要求 JWT 认证，因此所有请求都必须带上 Authorization 头。
token 保存在 gr.State 中，随每次交互传入。
"""

import json
import time

import gradio as gr
import requests

API_BASE = "http://localhost:8000/api/v1"
REQUEST_TIMEOUT = 30

NOT_LOGGED_IN = "⚠️ 请先登录"


def _headers(token):
    """构造认证请求头"""
    return {"Authorization": f"Bearer {token}"} if token else {}


def _error_message(resp):
    """从错误响应中提取可读信息"""
    if resp.status_code == 401:
        return "认证已失效，请重新登录"
    if resp.status_code == 403:
        return "权限不足"
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return f"HTTP {resp.status_code}"
    if isinstance(detail, dict):
        return detail.get("message", f"HTTP {resp.status_code}")
    return str(detail or f"HTTP {resp.status_code}")


def login(username, password):
    """登录并取得访问令牌"""
    if not username or not password:
        return "请输入用户名和密码", None
    try:
        resp = requests.post(
            f"{API_BASE}/auth/login",
            json={"username": username, "password": password},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            # 登录接口的 401 表示用户名或密码错误，而非令牌失效，
            # 因此不能复用 _error_message 的通用提示。
            try:
                detail = resp.json().get("detail")
            except ValueError:
                detail = None
            message = detail.get("message") if isinstance(detail, dict) else detail
            return f"❌ 登录失败：{message or f'HTTP {resp.status_code}'}", None
        payload = resp.json()
        user = payload.get("user") or {}
        return (
            f"✅ 已登录：{user.get('username', username)}（{user.get('role', '未知角色')}）",
            payload["access_token"],
        )
    except requests.RequestException as e:
        return f"❌ 无法连接后端：{e}", None


def upload_document(token, file, title, project, company):
    if not token:
        return NOT_LOGGED_IN, None
    if not file:
        return "请选择文件", None
    try:
        with open(file, "rb") as fh:
            resp = requests.post(
                f"{API_BASE}/documents/upload",
                files={"file": fh},
                data={
                    "title": title or "",
                    "project_name": project or "",
                    "company": company or "",
                },
                headers=_headers(token),
                timeout=120,
            )
        if resp.status_code != 200:
            return f"❌ 失败：{_error_message(resp)}", None
        doc_id = resp.json()["data"]["id"]
        return f"✅ 上传成功！ID: {doc_id}", doc_id
    except requests.RequestException as e:
        return f"❌ 失败：{e}", None
    except OSError as e:
        return f"❌ 无法读取文件：{e}", None


def create_task(token, doc_id):
    if not token:
        return NOT_LOGGED_IN, None
    if not doc_id:
        return "请先上传", None
    try:
        resp = requests.post(
            f"{API_BASE}/audit/tasks",
            params={"doc_id": doc_id},
            headers=_headers(token),
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return f"❌ 失败：{_error_message(resp)}", None
        task_id = resp.json()["data"]["task_id"]
        return f"✅ 任务已创建，ID: {task_id}", task_id
    except requests.RequestException as e:
        return f"❌ 失败：{e}", None


def poll_result(token, task_id):
    if not token:
        return NOT_LOGGED_IN, "{}"
    if not task_id:
        return "请先创建", "{}"
    for _ in range(30):
        try:
            resp = requests.get(
                f"{API_BASE}/audit/tasks/{task_id}",
                headers=_headers(token),
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                return f"❌ 查询失败：{_error_message(resp)}", "{}"
            data = resp.json()["data"]
            if data["status"] == "completed":
                result = requests.get(
                    f"{API_BASE}/audit/tasks/{task_id}/result",
                    headers=_headers(token),
                    timeout=REQUEST_TIMEOUT,
                )
                if result.status_code != 200:
                    return f"❌ 获取结果失败：{_error_message(result)}", "{}"
                return "✅ 完成！", json.dumps(result.json()["data"], indent=2, ensure_ascii=False)
            if data["status"] == "failed":
                return f"❌ 审核失败：{data.get('current_step') or '未知原因'}", "{}"
            time.sleep(1)
        except requests.RequestException as e:
            return f"❌ 网络错误：{e}", "{}"
    return "⏰ 超时", "{}"


def search_kb(token, q):
    if not token:
        return {"error": NOT_LOGGED_IN}
    if not q:
        return {"error": "请输入"}
    try:
        resp = requests.get(
            f"{API_BASE}/kb/clauses",
            params={"q": q, "top_k": 5},
            headers=_headers(token),
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return {"error": _error_message(resp)}
        return resp.json()["data"]
    except requests.RequestException as e:
        return {"error": str(e)}


def build_ui():
    with gr.Blocks(title="工安智审") as demo:
        gr.Markdown("# 🏗️ 工安智审 BuildSentry")

        # 令牌状态，随每次交互传入各回调
        token_state = gr.State()

        with gr.Row():
            username = gr.Textbox(label="用户名", scale=2)
            password = gr.Textbox(label="密码", type="password", scale=2)
            login_btn = gr.Button("🔑 登录", scale=1)
        login_status = gr.Textbox(label="登录状态", interactive=False)

        with gr.Tabs():
            with gr.TabItem("📄 新审核"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(label="选择文件", file_types=[".pdf", ".docx"])
                        title = gr.Textbox(label="标题", placeholder="可选")
                        project = gr.Textbox(label="项目")
                        company = gr.Textbox(label="单位")
                        upload_btn = gr.Button("📤 上传")
                        doc_status = gr.Textbox(label="状态", interactive=False)
                        doc_id_state = gr.State()
                    with gr.Column():
                        task_btn = gr.Button("🚀 创建审核")
                        task_status = gr.Textbox(label="任务状态", interactive=False)
                        task_id_state = gr.State()
                        poll_btn = gr.Button("📊 获取结果")
                        result_output = gr.JSON(label="报告", height=500)

                login_btn.click(login, [username, password], [login_status, token_state])
                upload_btn.click(
                    upload_document,
                    [token_state, file_input, title, project, company],
                    [doc_status, doc_id_state],
                )
                task_btn.click(
                    create_task, [token_state, doc_id_state], [task_status, task_id_state]
                )
                poll_btn.click(
                    poll_result, [token_state, task_id_state], [task_status, result_output]
                )

            with gr.TabItem("📚 检索"):
                query = gr.Textbox(label="查询")
                search_btn = gr.Button("🔍")
                output = gr.JSON()
                search_btn.click(search_kb, [token_state, query], [output])

    return demo


if __name__ == "__main__":
    # 容器部署需要监听所有网卡才能被外部访问（与 Dockerfile 中 uvicorn 的做法一致）
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)  # noqa: S104
