# ModelScope Image CLI (造相-Z-Image)

一个面向 **ModelScope 造相-Z-Image / Z-Image-Turbo** 的轻量命令行工具，支持异步文生图、批量任务、比例映射分辨率、自动轮询与下载。

脚本入口：`generate_images_cli.py`

## 详细教程与示例

- 基于这个脚本的详细使用教程与示例：
  https://mp.weixin.qq.com/s/Z3UroXCMMHrab8s0CIap9A

## 功能特性

- 支持 **造相-Z-Image / Z-Image-Turbo**，也支持自定义 `--model`
- 支持批量生成（单任务 `n` 或多任务循环）
- 支持 `--ratio`（如 `3:4`、`16:9`）自动映射分辨率
- 支持 `--size` / `--width --height` 两套分辨率参数策略
- 支持负向提示词、`seed`、`steps`、`guidance_scale`、LoRA
- 自动轮询任务状态并下载图片到本地
- 可选 `--check-size` 输出落盘图片尺寸

## 环境要求

- Python 3.10+
- 依赖：
  - `requests`（必需）
  - `Pillow`（仅 `--check-size` 时需要）

安装示例：

```bash
python3 -m venv venv
source venv/bin/activate
pip install requests pillow
```

## 快速开始

### 1) 默认模型（Z-Image-Turbo）按比例批量生成

```bash
python generate_images_cli.py \
  --prompt "一位年轻女性在图书馆读书，柔和的自然光，35mm 胶片摄影风格" \
  --num-images 10 \
  --ratio 3:4 \
  --check-size
```

### 2) 指定模型为造相-Z-Image

```bash
python generate_images_cli.py \
  --model Tongyi-MAI/Z-Image \
  --prompt "cinematic portrait, soft light" \
  --num-images 4 \
  --ratio 9:16
```

### 3) 显式指定尺寸

```bash
python generate_images_cli.py \
  --prompt "minimalist poster" \
  --size 1024x1536 \
  --num-images 2
```

### 4) 打印请求 payload（便于调试）

```bash
python generate_images_cli.py \
  --prompt "test prompt" \
  --ratio 1:1 \
  --print-payload
```

## 常用参数

- `--prompt`：提示词（必填）
- `--api-key`：ModelScope Token
- `--model`：模型 ID（默认 `Tongyi-MAI/Z-Image-Turbo`）
- `--base-url`：API 地址（默认 `https://api-inference.modelscope.cn/`）
- `--num-images`：目标图片数量
- `--mode`：批量策略
  - `auto`：先尝试 API 批量，再在必要时回退循环
  - `api`：单任务请求 `n`
  - `loop`：循环提交多个任务
- `--ratio`：宽高比（如 `1:1`、`3:4`、`16:9`）
- `--size`：尺寸（支持 `WIDTHxHEIGHT` 或 `WIDTH*HEIGHT` 输入）
- `--width` / `--height`：显式宽高（需同时提供）
- `--resolution-mode`：分辨率字段发送策略
  - `auto`：`Tongyi-MAI/Z-Image*` 用 `size`，其他模型用 `width/height`
  - `size`：仅发送 `size`（格式 `WxH`）
  - `width-height`：仅发送 `width` 与 `height`
  - `both`：同时发送 `size` + `width` + `height`
- `--negative-prompt`：负向提示词
- `--seed` / `--steps` / `--guidance-scale`：生成参数
- `--lora`：LoRA 配置
- `--extra-json`：透传额外 JSON 字段
- `--output-dir`：输出目录（默认 `result_pic`）
- `--filename-prefix`：输出文件名前缀
- `--check-size`：保存后打印图片尺寸

## 输出目录结构

图片默认保存到：

```text
result_pic/<timestamp>_<taskSuffixStart>-<taskSuffixEnd>/result_image_001.png
```

示例：

```text
result_pic/202602111735_1234-1243/result_image_001.png
```

## 说明

在部分模型链路里，`width/height` 可能不生效，`size=WxH` 更稳定。脚本在 `--resolution-mode auto` 下已针对 `Tongyi-MAI/Z-Image*` 做自动适配。

## License

本项目使用 [MIT License](./LICENSE)。
