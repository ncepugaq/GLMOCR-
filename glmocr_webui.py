# -*- coding: utf-8 -*-

import os
import sys
import re
import json
import argparse
import tempfile
import time
import traceback

def _global_excepthook(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    input("\nERROR - Press Enter to exit...")

sys.excepthook = _global_excepthook

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import gradio as gr
from PIL import Image, ImageOps
import fitz

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(PROJECT_DIR, "results_glmocr")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = "ZhipuAI/GLM-OCR"
LOCAL_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "GLM-OCR")

model = None
processor = None
stop_flag = False

OCR_MODES = {
    "通用文字识别 (Markdown)": "Recognize the text in the image and output in Markdown format.",
    "纯文本识别": "Text Recognition:",
    "表格识别": "表格识别:",
    "公式识别": "公式识别:",
}

KB_CHUNK_SIZE = 1000


def find_model_path():
    for path in [LOCAL_MODEL_PATH, MODEL_PATH]:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return path
    if os.path.isdir(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    return MODEL_PATH


def chunk_text(text, chunk_size=KB_CHUNK_SIZE, overlap=200):
    paragraphs = text.split('\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + '\n'
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para + '\n'
    if current.strip():
        chunks.append(current.strip())
    return chunks


def stop_processing():
    global stop_flag
    stop_flag = True
    return "[停止] 已发送停止信号，正在保存已完成页面..."


def load_model(model_path_override=None):
    global model, processor

    if model is not None:
        return "模型已加载"

    try:
        from modelscope import AutoProcessor, AutoModelForImageTextToText

        if model_path_override and model_path_override.strip():
            load_path = model_path_override.strip()
        else:
            load_path = find_model_path()

        status = f"正在从 {load_path} 加载 processor..."
        print(status)
        processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)

        status = f"正在加载 GLM-OCR 模型 (0.9B，BF16)..."
        print(status)
        model = AutoModelForImageTextToText.from_pretrained(
            load_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            torch.cuda.empty_cache()
            status = f"[OK] GLM-OCR 加载成功！GPU: {gpu_name} ({gpu_mem:.1f}GB)"
        else:
            status = "[OK] GLM-OCR 加载成功！CPU (推理速度较慢)"
        print(status)
        return status

    except Exception as e:
        error_msg = f"模型加载失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg


MAX_IMAGE_SIZE = 1024
MAX_NEW_TOKENS = 2048


def _prepare_image(image):
    if isinstance(image, str):
        pil_image = Image.open(image).convert('RGB')
    else:
        pil_image = image.convert('RGB')

    pil_image = ImageOps.exif_transpose(pil_image)
    w, h = pil_image.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
    return pil_image


def run_ocr(image, mode_key):
    if model is None:
        return "模型未加载，请先点击「加载模型」按钮！", None, None

    if image is None:
        return "请上传图片", None, None

    try:
        pil_image = _prepare_image(image)
        prompt = OCR_MODES.get(mode_key, OCR_MODES["通用文字识别 (Markdown)"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        inputs.pop("token_type_ids", None)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        elapsed = time.perf_counter() - t0

        new_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
        tps = new_tokens / elapsed if elapsed > 0 else 0
        print(f"[OCR] {new_tokens} tokens / {elapsed:.2f}s = {tps:.1f} tok/s")

        output_text = processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        del inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_filepath = os.path.join(RESULTS_DIR, f"glmocr_{timestamp}.md")
        with open(result_filepath, 'w', encoding='utf-8') as f:
            f.write(output_text)

        chunks = chunk_text(output_text)
        jsonl_filepath = os.path.join(RESULTS_DIR, f"glmocr_{timestamp}.jsonl")
        with open(jsonl_filepath, 'w', encoding='utf-8') as f:
            for ci, chunk in enumerate(chunks):
                record = {
                    "source": "单张图片",
                    "page": 1,
                    "chunk_id": ci,
                    "total_chunks": len(chunks),
                    "text": chunk,
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return output_text, result_filepath, jsonl_filepath

    except Exception as e:
        error_msg = f"OCR 处理失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None, None


def ocr_single_image(image, mode_key, prompt=None):
    if prompt is None:
        prompt = OCR_MODES.get(mode_key, OCR_MODES["通用文字识别 (Markdown)"])

    pil_image = _prepare_image(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    t0 = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    elapsed = time.perf_counter() - t0

    new_tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    tps = new_tokens / elapsed if elapsed > 0 else 0
    print(f"[OCR-pdf] {new_tokens} tokens / {elapsed:.2f}s = {tps:.1f} tok/s")

    output_text = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    del inputs, generated_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_text


def run_ocr_pdf(pdf_file, mode_key, dpi_scale, start_page, end_page, progress=gr.Progress()):
    global stop_flag
    stop_flag = False

    if model is None:
        return "模型未加载，请先点击「加载模型」按钮！", None, None

    if pdf_file is None:
        return "请上传PDF文件", None, None

    try:
        pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)

        if start_page < 1:
            start_page = 1
        if end_page < 1 or end_page > total_pages:
            end_page = total_pages
        if start_page > end_page:
            return "起始页不能大于结束页！", None, None

        start_page = int(start_page)
        end_page = int(end_page)
        dpi_scale = float(dpi_scale)

        pages_to_process = list(range(start_page - 1, end_page))
        num_pages = len(pages_to_process)

        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_basename = re.sub(r'[^\w\-.]', '_', pdf_basename)

        output_dir = os.path.join(RESULTS_DIR, f"{pdf_basename}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)

        all_results = []
        status_msgs = []
        total_time = 0
        all_kb_records = []

        prompt = OCR_MODES.get(mode_key, OCR_MODES["通用文字识别 (Markdown)"])

        for idx, page_num in enumerate(pages_to_process):
            if stop_flag:
                status_msgs.append("[中断] 用户中断")
                break

            page_start = time.time()
            page_display = page_num + 1

            progress(idx / num_pages, desc=f"处理第 {page_display}/{end_page} 页 | 总进度")

            page = pdf_doc[page_num]
            mat = fitz.Matrix(dpi_scale / 72, dpi_scale / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_path = os.path.join(output_dir, f"_page_{page_display:04d}.png")
            pix.save(img_path)

            pil_img = Image.open(img_path).convert('RGB')
            page_text = ocr_single_image(pil_img, mode_key, prompt)

            all_results.append(f"## 第 {page_display} 页\n\n{page_text}\n\n---\n\n")

            chunks = chunk_text(page_text)
            for ci, chunk in enumerate(chunks):
                all_kb_records.append({
                    "source": f"{pdf_basename}.pdf",
                    "page": page_display,
                    "total_pages": total_pages,
                    "chunk_id": ci,
                    "total_chunks_in_page": len(chunks),
                    "text": chunk,
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "ocr_mode": mode_key,
                })

            page_time = time.time() - page_start
            total_time += page_time
            avg_time = total_time / (idx + 1)
            remaining = avg_time * (num_pages - idx - 1)
            status_msgs.append(
                f"第 {page_display} 页完成 ({page_time:.1f}s) | {len(chunks)} 个分块"
            )

            try:
                os.remove(img_path)
            except Exception:
                pass

        pdf_doc.close()
        actual_pages = len(all_results)
        progress(1.0, desc="正在保存结果...")

        if stop_flag and actual_pages < num_pages:
            combined = f"# PDF OCR 识别结果（部分，已中断）\n\n"
            combined += f"- 总页数: {total_pages}\n"
            combined += f"- 已完成: {actual_pages}/{num_pages}\n"
            combined += f"- 总耗时: {total_time:.1f}s\n"
            combined += f"- 平均每页: {total_time / actual_pages:.1f}s\n\n---\n\n"
        else:
            combined = f"# PDF OCR 识别结果\n\n"
            combined += f"- 总页数: {total_pages}\n"
            combined += f"- 处理页数: {num_pages} (第 {start_page}-{end_page} 页)\n"
            combined += f"- 总耗时: {total_time:.1f}s\n"
            combined += f"- 平均每页: {total_time / num_pages:.1f}s\n\n---\n\n"

        combined += "".join(all_results)

        result_file = os.path.join(output_dir, f"{pdf_basename}_ocr.md")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(combined)

        pages_dir = os.path.join(output_dir, "pages")
        os.makedirs(pages_dir, exist_ok=True)
        for i, page_content in enumerate(all_results):
            page_num = start_page + i
            page_file = os.path.join(pages_dir, f"page_{page_num:04d}.md")
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(page_content)

        jsonl_file = os.path.join(output_dir, f"{pdf_basename}_kb.jsonl")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for record in all_kb_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = "\n".join(status_msgs)
        if stop_flag and actual_pages < num_pages:
            status += f"\n\n[中断] 用户中断！已完成 {actual_pages}/{num_pages} 页"
        else:
            status += f"\n\n[完成] 处理完成！共 {num_pages} 页，耗时 {total_time:.1f}s"
            status += f"\n平均每页 {total_time / num_pages:.1f}s"
            status += f"\n共 {len(all_kb_records)} 条知识库分块"

        return status, result_file, jsonl_file

    except Exception as e:
        error_msg = f"PDF处理失败: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg, None, None


def create_ui():
    with gr.Blocks(title="GLM-OCR 文档识别") as demo:
        gr.Markdown("# GLM-OCR 文档识别", elem_classes="main-title")
        gr.Markdown("智谱 AI GLM-OCR (0.9B) | 支持文字/表格/公式识别 | 单图 & PDF 批量处理", elem_classes="subtitle")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 模型配置")
                load_btn = gr.Button("加载模型", variant="primary", size="lg")
                model_status = gr.Textbox(label="模型状态", value="模型尚未加载", interactive=False)

                with gr.Accordion("高级设置", open=False):
                    model_path_input = gr.Textbox(
                        label="模型路径 (留空使用默认)",
                        placeholder="ZhipuAI/GLM-OCR",
                        info="留空则自动查找本地路径或从 ModelScope 下载"
                    )

                ocr_mode_input = gr.Radio(
                    label="识别模式",
                    choices=list(OCR_MODES.keys()),
                    value="通用文字识别 (Markdown)",
                    info="通用 Markdown: 文字+表格+公式自动识别 | 纯文本: 仅输出文字"
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("单张图片"):
                        gr.Markdown("### 上传图片")
                        input_image = gr.Image(label="上传图片", type="pil", height=350)
                        run_btn = gr.Button("开始识别", variant="primary", size="lg")

                        gr.Markdown("### 识别结果")
                        with gr.Tabs():
                            with gr.TabItem("格式化结果"):
                                output_md = gr.Markdown(label="识别结果", show_label=False)
                            with gr.TabItem("原始输出"):
                                output_raw = gr.Textbox(label="原始输出", lines=15, max_lines=30)

                        with gr.Row():
                            output_download = gr.File(label="下载 Markdown")
                            output_jsonl = gr.File(label="下载 JSONL")

                    with gr.TabItem("PDF 批量处理"):
                        gr.Markdown("### 上传 PDF")
                        pdf_file = gr.File(label="选择 PDF 文件", file_types=[".pdf"])
                        pdf_status = gr.Textbox(label="处理状态", value="等待上传PDF...", lines=4, interactive=False)

                        with gr.Row():
                            pdf_dpi = gr.Slider(
                                label="渲染 DPI", minimum=72, maximum=300, value=100, step=2,
                                info="推荐 100 DPI (高清用 150-200，速度优先用 100)"
                            )
                        with gr.Row():
                            pdf_start = gr.Number(label="起始页", value=1, precision=0, minimum=1)
                            pdf_end = gr.Number(label="结束页 (0=最后一页)", value=0, precision=0, minimum=0)

                        pdf_run_btn = gr.Button("开始批量 OCR", variant="primary", size="lg")
                        with gr.Row():
                            pdf_stop_btn = gr.Button("停止处理", variant="stop", size="sm")
                            pdf_stop_status = gr.Textbox(label="", value="", interactive=False, show_label=False, container=False)

                        gr.Markdown(
                            "> **提示**：处理过程中请勿关闭浏览器标签页。"
                            "结果自动保存到 `E:\\OCR\\results_glmocr\\` 目录。",
                            elem_classes="subtitle"
                        )

                        gr.Markdown("### 结果")
                        with gr.Row():
                            pdf_download = gr.File(label="下载 Markdown (全文)")
                            pdf_jsonl = gr.File(label="下载 JSONL (知识库分块)")

        load_btn.click(
            fn=lambda path: load_model(path if path and path.strip() else None),
            inputs=[model_path_input],
            outputs=[model_status],
        )

        run_btn.click(
            fn=run_ocr,
            inputs=[input_image, ocr_mode_input],
            outputs=[output_raw, output_download, output_jsonl],
        )

        pdf_run_btn.click(
            fn=run_ocr_pdf,
            inputs=[pdf_file, ocr_mode_input, pdf_dpi, pdf_start, pdf_end],
            outputs=[pdf_status, pdf_download, pdf_jsonl],
        )

        pdf_stop_btn.click(
            fn=stop_processing,
            inputs=[],
            outputs=[pdf_stop_status],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--auto-load", action="store_true", help="启动时自动加载模型")
    args = parser.parse_args()

    if args.auto_load:
        print("正在自动加载模型...")
        load_model()

    demo = create_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        """,
    )
