# =============================================================================
#  new_webui.py  —  AI Editing Integrated Platform
#
#  Run:   set HF_HUB_OFFLINE=1 && uv run new_webui.py --port 7861
#
#  Flow:  Select/upload avatar + text → one click → talking-head video
#         Avatars auto-saved after first successful generation.
#         DeepSeek AI can generate speaking scripts from brief ideas.
#         Full Chinese/English UI toggle (title always English).
# =============================================================================

import html
import json
import os
import sys
import threading
import time
import subprocess
import uuid as _uuid

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="AI Editing Integrated Platform",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--model_dir", type=str, default="./checkpoints")
parser.add_argument("--fp16", action="store_true", default=False)
parser.add_argument("--deepspeed", action="store_true", default=False)
parser.add_argument("--cuda_kernel", action="store_true", default=False)
parser.add_argument("--gui_seg_tokens", type=int, default=120)
parser.add_argument("--duix_url", type=str, default="http://127.0.0.1:8384/easy")
parser.add_argument("--duix_data_root", type=str, default="")
parser.add_argument("--deepseek_api_key", type=str, default="",
                    help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist.")
    sys.exit(1)

for file in ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

from duix_integration import (
    extract_audio_from_video,
    submit_video_synthesis,
    poll_video_result,
    get_avatar_choices,
    load_avatar_by_name,
    save_avatar,
    delete_avatar,
    _load_avatars,
    generate_prompt_text,
)
import duix_integration as _duix

_duix.DUIX_BASE_URL   = cmd_args.duix_url
_duix.DUIX_SUBMIT_URL = f"{cmd_args.duix_url}/submit"
_duix.DUIX_QUERY_URL  = f"{cmd_args.duix_url}/query"
if cmd_args.duix_data_root:
    _duix.DUIX_HOST_DATA_ROOT = cmd_args.duix_data_root
    _duix.DUIX_HOST_TEMP = os.path.join(cmd_args.duix_data_root, "temp")

if cmd_args.deepseek_api_key:
    _duix.DEEPSEEK_API_KEY = cmd_args.deepseek_api_key

print(f"[DuiX] API endpoint  : {_duix.DUIX_SUBMIT_URL}")
print(f"[DuiX] Shared volume : {_duix.DUIX_HOST_DATA_ROOT}")
print(f"[DuiX] Container root: {_duix.DUIX_CONTAINER_DATA_ROOT}")
print(f"[DeepSeek] API key   : {'***' + _duix.DEEPSEEK_API_KEY[-4:] if _duix.DEEPSEEK_API_KEY else 'NOT SET'}")

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_fp16=cmd_args.fp16,
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
)

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("outputs/videos", exist_ok=True)
os.makedirs("outputs/extracted_audio", exist_ok=True)
os.makedirs("avatars", exist_ok=True)


# =========================================================================
# Bilingual UI labels — Chinese default
# =========================================================================

UI_LABELS = {
    "zh": {
        "tab_generate": "视频生成",
        "source_video": "📹 源视频",
        "ref_audio": "🎤 参考音频（克隆声音）",
        "text_to_speak": "📝 要说的文本",
        "text_placeholder": "请输入目标文本",
        "ai_script_idea": "🤖 AI 脚本创意",
        "ai_placeholder": "例如：'介绍我们的新AI产品' 或 '欢迎来到直播间'",
        "btn_ai_generate": "✨ 生成脚本",
        "btn_generate_video": "🎬 生成视频",
        "result_video": "🎬 生成结果",
        "generated_audio": "🔊 生成音频",
        "saved_avatars": "👤 已保存的数字人",
        "no_avatars": "*暂无已保存的数字人。生成视频后将自动保存。*",
        "btn_use": "✅ 使用",
        "lang_toggle": "🌐 English",
        "show_experimental": "显示实验功能",
        "glossary_toggle": "开启术语词汇读音",
        "feature_settings": "功能设置",
        "emo_control_method": "情感控制方式",
        "emo_same_as_ref": "与音色参考音频相同",
        "emo_use_ref_audio": "使用情感参考音频",
        "emo_use_vector": "使用情感向量控制",
        "emo_use_text": "使用情感描述文本控制",
        "emo_ref_upload": "上传情感参考音频",
        "emo_random_sample": "情感随机采样",
        "emo_weight": "情感权重",
        "emo_text_label": "情感描述文本",
        "emo_text_placeholder": "请输入情绪描述（或留空以自动使用目标文本作为情绪描述）",
        "emo_text_info": "例如：委屈巴巴、危险在悄悄逼近",
        "experimental_warning": "提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。",
        "vec_joy": "喜",
        "vec_anger": "怒",
        "vec_sadness": "哀",
        "vec_fear": "惧",
        "vec_disgust": "厌恶",
        "vec_depressed": "低落",
        "vec_surprise": "惊喜",
        "vec_calm": "平静",
        "glossary_title": "自定义术语词汇读音",
        "glossary_desc": "自定义个别专业术语的读音",
        "glossary_term": "术语",
        "glossary_zh_reading": "中文读法",
        "glossary_en_reading": "英文读法",
        "btn_add_term": "添加术语",
        "no_glossary": "暂无术语",
        "advanced_settings": "高级生成参数设置",
        "gpt2_sampling": "GPT2 采样设置",
        "gpt2_sampling_desc": "参数会影响音频多样性和生成速度详见",
        "do_sample_info": "是否进行采样",
        "segment_settings": "分句设置",
        "segment_settings_desc": "参数会影响音频质量和生成速度",
        "max_segment_tokens": "分句最大Token数",
        "max_segment_info": "建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高",
        "preview_segments": "预览分句结果",
        "col_index": "序号",
        "col_content": "分句内容",
        "col_tokens": "Token数",
        "max_mel_info": "生成Token最大数量，过小导致音频被截断",
        "msg_upload_video": "⚠️ 请先上传源视频",
        "msg_upload_audio": "⚠️ 请上传参考音频用于声音克隆",
        "msg_enter_text": "⚠️ 请输入要说的文本",
        "msg_step1": "步骤 1/2 — 正在生成语音...",
        "msg_step2_submit": "步骤 2/2 — 正在提交到DuiX...",
        "msg_step2_wait": "步骤 2/2 — 等待DuiX合成...",
        "msg_step2_duix": "步骤 2/2 — DuiX: ",
        "msg_audio_done": "🔊 语音已生成，开始合成视频...",
        "msg_done": "✅ 完成！",
        "msg_video_done": "✅ 视频合成完成！",
        "msg_enter_idea": "⚠️ 请先输入主题或创意",
        "msg_generating_script": "🤖 正在用AI生成脚本...",
        "msg_script_done": "✅ 脚本生成成功！",
        "msg_audio_extracted": "✅ 已从视频中提取音频 → 自动填入参考音频",
        "msg_no_audio_track": "ℹ️ 视频中未找到音频，请手动上传参考音频",
        "msg_avatar_loaded": "👤 已加载数字人: ",
        "msg_avatar_deleted": "🗑️ 已删除数字人: ",
        "msg_avatar_name": "数字人",
        "msg_enter_term": "请输入术语",
        "msg_enter_reading": "请至少输入一种读法",
        "msg_glossary_updated": "词汇表已更新",
        "msg_glossary_error": "保存词汇表时出错",
        "avatar_dropdown_label": "选择已保存的数字人",
        "avatar_dropdown_placeholder": "-- 点击选择 --",
        "btn_load_avatar": "📂 加载",
        "btn_delete_avatar": "🗑️ 删除",
        "btn_refresh_avatars": "🔄",
    },
    "en": {
        "tab_generate": "Generate",
        "source_video": "📹 Source Video",
        "ref_audio": "🎤 Reference Audio (voice to clone)",
        "text_to_speak": "📝 Text to Speak",
        "text_placeholder": "Enter the text you want spoken",
        "ai_script_idea": "🤖 AI Script Idea",
        "ai_placeholder": "e.g. 'introduce our new AI product' or 'welcome to our livestream'",
        "btn_ai_generate": "✨ Generate",
        "btn_generate_video": "🎬 Generate Video",
        "result_video": "🎬 Result Video",
        "generated_audio": "🔊 Generated Audio",
        "saved_avatars": "👤 Saved Avatars",
        "no_avatars": "*No saved avatars yet. Generate a video to save one automatically.*",
        "btn_use": "✅ Use",
        "lang_toggle": "🌐 中文",
        "show_experimental": "Show Experimental",
        "glossary_toggle": "Enable Glossary",
        "feature_settings": "Feature Settings",
        "emo_control_method": "Emotion Control Method",
        "emo_same_as_ref": "Same as voice reference",
        "emo_use_ref_audio": "Use emotion reference audio",
        "emo_use_vector": "Use emotion vector",
        "emo_use_text": "Use emotion description text",
        "emo_ref_upload": "Upload emotion reference audio",
        "emo_random_sample": "Random emotion sampling",
        "emo_weight": "Emotion Weight",
        "emo_text_label": "Emotion Description Text",
        "emo_text_placeholder": "Enter emotion description (or leave blank to auto-use target text)",
        "emo_text_info": "e.g.: excited, calm and soothing, urgent warning",
        "experimental_warning": "Note: This feature is experimental. Results may vary. We are continuously improving.",
        "vec_joy": "Joy",
        "vec_anger": "Anger",
        "vec_sadness": "Sadness",
        "vec_fear": "Fear",
        "vec_disgust": "Disgust",
        "vec_depressed": "Depressed",
        "vec_surprise": "Surprise",
        "vec_calm": "Calm",
        "glossary_title": "Custom Term Pronunciation",
        "glossary_desc": "Customize pronunciation for specific terms",
        "glossary_term": "Term",
        "glossary_zh_reading": "Chinese Reading",
        "glossary_en_reading": "English Reading",
        "btn_add_term": "Add Term",
        "no_glossary": "No terms yet",
        "advanced_settings": "Advanced Generation Settings",
        "gpt2_sampling": "GPT2 Sampling Settings",
        "gpt2_sampling_desc": "Parameters affect audio diversity and generation speed. See",
        "do_sample_info": "Whether to sample",
        "segment_settings": "Segmentation Settings",
        "segment_settings_desc": "Parameters affect audio quality and generation speed",
        "max_segment_tokens": "Max Tokens per Segment",
        "max_segment_info": "Recommended 80-200. Larger = longer segments; smaller = more fragments",
        "preview_segments": "Preview Segments",
        "col_index": "Index",
        "col_content": "Segment Content",
        "col_tokens": "Tokens",
        "max_mel_info": "Max generated tokens. Too small will truncate audio",
        "msg_upload_video": "⚠️ Upload a source video first.",
        "msg_upload_audio": "⚠️ Upload a reference audio for voice cloning.",
        "msg_enter_text": "⚠️ Enter the text you want spoken.",
        "msg_step1": "Step 1/2 — Generating speech audio…",
        "msg_step2_submit": "Step 2/2 — Submitting to DuiX…",
        "msg_step2_wait": "Step 2/2 — Waiting for DuiX…",
        "msg_step2_duix": "Step 2/2 — DuiX: ",
        "msg_audio_done": "🔊 Speech audio generated. Starting video synthesis…",
        "msg_done": "✅ Done!",
        "msg_video_done": "✅ Video synthesis complete!",
        "msg_enter_idea": "⚠️ Enter a topic or idea first.",
        "msg_generating_script": "🤖 Generating script with DeepSeek...",
        "msg_script_done": "✅ Script generated!",
        "msg_audio_extracted": "✅ Audio extracted from video → auto-filled as reference audio.",
        "msg_no_audio_track": "ℹ️ No audio track found. Upload reference audio manually.",
        "msg_avatar_loaded": "👤 Loaded avatar: ",
        "msg_avatar_deleted": "🗑️ Deleted avatar: ",
        "msg_avatar_name": "Avatar",
        "msg_enter_term": "Please enter a term",
        "msg_enter_reading": "Enter at least one reading",
        "msg_glossary_updated": "Glossary updated",
        "msg_glossary_error": "Error saving glossary",
        "avatar_dropdown_label": "Select saved avatar",
        "avatar_dropdown_placeholder": "-- Click to select --",
        "btn_load_avatar": "📂 Load",
        "btn_delete_avatar": "🗑️ Delete",
        "btn_refresh_avatars": "🔄",
    },
}

_current_lang = "zh"


def L(key: str) -> str:
    return UI_LABELS.get(_current_lang, UI_LABELS["zh"]).get(key, key)


# =========================================================================
# Helper functions
# =========================================================================

def format_glossary_markdown():
    if not tts.normalizer.term_glossary:
        return L("no_glossary")
    lines = [f"| {L('glossary_term')} | {L('glossary_zh_reading')} | {L('glossary_en_reading')} |",
             "|---|---|---|"]
    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")
    return "\n".join(lines)


def run_tts(prompt, text, emo_control_method, emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random, max_text_tokens_per_segment,
            *args):
    output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        vec = None
    if emo_text == "":
        emo_text = None
    result = tts.infer(
        spk_audio_prompt=prompt, text=text, output_path=output_path,
        emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
        emo_vector=vec,
        use_emo_text=(emo_control_method == 3), emo_text=emo_text,
        use_random=emo_random, verbose=cmd_args.verbose,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        **kwargs,
    )
    return result


def update_prompt_audio():
    return gr.update(interactive=True)


# =========================================================================
# Language toggle handler
# =========================================================================

def on_lang_toggle(current_lang_state):
    global _current_lang
    new_lang = "en" if current_lang_state == "zh" else "zh"
    _current_lang = new_lang
    t = UI_LABELS[new_lang]

    emo_choices = [t["emo_same_as_ref"], t["emo_use_ref_audio"], t["emo_use_vector"]]

    experimental_warning_html = (
        f'<div style="padding:0.5em 0.8em;border-radius:0.5em;'
        f'background:#ffa87d;color:#000;font-weight:bold">'
        f'{html.escape(t["experimental_warning"])}</div>'
    )

    return (
        new_lang,                                                          # lang_state
        gr.update(label=t["source_video"]),                                # video_upload
        gr.update(label=t["ref_audio"]),                                   # prompt_audio
        gr.update(label=t["text_to_speak"],
                  placeholder=t["text_placeholder"]),                      # input_text_single
        gr.update(label=t["ai_script_idea"],
                  placeholder=t["ai_placeholder"]),                        # prompt_idea_input
        gr.update(value=t["btn_ai_generate"]),                             # prompt_gen_btn
        gr.update(value=t["btn_generate_video"]),                          # generate_btn
        gr.update(label=t["result_video"]),                                # output_video
        gr.update(value=t["lang_toggle"]),                                 # lang_btn
        gr.update(label=t["show_experimental"]),                           # experimental_checkbox
        gr.update(label=t["glossary_toggle"]),                             # glossary_checkbox
        gr.update(label=t["feature_settings"]),                            # feature_accordion
        gr.update(label=t["emo_control_method"],
                  choices=emo_choices, value=emo_choices[0]),               # emo_control_method
        gr.update(label=t["emo_ref_upload"]),                              # emo_upload
        gr.update(label=t["emo_random_sample"]),                           # emo_random
        gr.update(label=t["emo_weight"]),                                  # emo_weight
        gr.update(label=t["emo_text_label"],
                  placeholder=t["emo_text_placeholder"],
                  info=t["emo_text_info"]),                                # emo_text
        experimental_warning_html,                                         # emo_warning_html
        gr.update(label=t["vec_joy"]),                                     # vec1
        gr.update(label=t["vec_anger"]),                                   # vec2
        gr.update(label=t["vec_sadness"]),                                 # vec3
        gr.update(label=t["vec_fear"]),                                    # vec4
        gr.update(label=t["vec_disgust"]),                                 # vec5
        gr.update(label=t["vec_depressed"]),                               # vec6
        gr.update(label=t["vec_surprise"]),                                # vec7
        gr.update(label=t["vec_calm"]),                                    # vec8
        gr.update(label=t["glossary_title"]),                              # glossary_accordion
        gr.update(value=t["glossary_desc"]),                               # glossary_desc_md
        gr.update(label=t["glossary_term"]),                               # glossary_term
        gr.update(label=t["glossary_zh_reading"]),                         # glossary_reading_zh
        gr.update(label=t["glossary_en_reading"]),                         # glossary_reading_en
        gr.update(value=t["btn_add_term"]),                                # btn_add_term
        gr.update(value=format_glossary_markdown()),                       # glossary_table
        gr.update(label=t["advanced_settings"]),                           # advanced_accordion
        gr.update(info=t["do_sample_info"]),                               # do_sample
        gr.update(info=t["max_mel_info"]),                                 # max_mel_tokens
        gr.update(label=t["max_segment_tokens"],
                  info=t["max_segment_info"]),                             # max_text_tokens_per_segment
        gr.update(label=t["preview_segments"]),                            # preview_accordion
        gr.update(label=t["saved_avatars"]),                               # avatar_accordion
        gr.update(label=t["avatar_dropdown_label"],
                  choices=get_avatar_choices(),
                  value=None),                                             # avatar_dropdown
        gr.update(value=t["btn_load_avatar"]),                             # btn_load_avatar
        gr.update(value=t["btn_delete_avatar"]),                           # btn_delete_avatar
    )


# =========================================================================
# DeepSeek prompt generation handler
# =========================================================================

def on_generate_prompt(idea_text):
    if not idea_text or not idea_text.strip():
        gr.Warning(L("msg_enter_idea"))
        return gr.update()
    gr.Info(L("msg_generating_script"))
    result = generate_prompt_text(idea_text.strip())
    if result["success"]:
        gr.Info(L("msg_script_done"))
        return gr.update(value=result["text"])
    else:
        gr.Warning(f"⚠️ {result['message']}")
        return gr.update()


# =========================================================================
# Avatar handlers (dropdown-based — reliable)
# =========================================================================

def _load_avatars_safe():
    try:
        return _load_avatars()
    except Exception:
        return []


def _get_avatar_dropdown_choices():
    """Build dropdown choices: 'name (date)' list."""
    avatars = _load_avatars_safe()
    return [a["name"] for a in avatars]


def on_avatar_refresh():
    """Refresh the avatar dropdown."""
    choices = _get_avatar_dropdown_choices()
    if not choices:
        gr.Info(L("no_avatars"))
    return gr.update(choices=choices, value=None)


def on_avatar_load(selected_name):
    """Load selected avatar into video + audio."""
    if not selected_name:
        gr.Warning("⚠️ Select an avatar first.")
        return gr.update(), gr.update(), ""

    avatar = load_avatar_by_name(selected_name)
    if not avatar:
        gr.Warning(f"Avatar '{selected_name}' not found.")
        return gr.update(), gr.update(), ""

    vid = avatar.get("video_path", "")
    aud = avatar.get("audio_path", "")
    vid_ok = vid and os.path.isfile(vid)
    aud_ok = aud and os.path.isfile(aud)

    if not vid_ok:
        gr.Warning(f"Video file missing for '{selected_name}'.")
    if not aud_ok:
        gr.Warning(f"Audio file missing for '{selected_name}'.")

    gr.Info(f"{L('msg_avatar_loaded')}{selected_name}")

    return (
        gr.update(value=vid if vid_ok else None),
        gr.update(value=aud if aud_ok else None),
        vid if vid_ok else "",
    )


def on_avatar_delete(selected_name):
    """Delete selected avatar and refresh dropdown."""
    if not selected_name:
        gr.Warning("⚠️ Select an avatar first.")
        return gr.update(), gr.update(), gr.update(), ""

    delete_avatar(selected_name)
    gr.Info(f"{L('msg_avatar_deleted')}{selected_name}")

    choices = _get_avatar_dropdown_choices()
    return (
        gr.update(choices=choices, value=None),
        gr.update(value=None),
        gr.update(value=None),
        "",
    )


def auto_save_avatar(video_path, audio_path):
    """Auto-save avatar after successful generation."""
    if not video_path or not audio_path:
        return gr.update()
    if not os.path.isfile(video_path) or not os.path.isfile(audio_path):
        return gr.update()
    name = f"{L('msg_avatar_name')} {time.strftime('%m/%d %H:%M')}"
    save_avatar(name, video_path, audio_path)
    choices = _get_avatar_dropdown_choices()
    return gr.update(choices=choices, value=name)


# =========================================================================
# Video upload + Generate handlers
# =========================================================================

def on_video_upload(video_path):
    if not video_path:
        return gr.update(), ""
    print(f"[DuiX] Video uploaded: {video_path}")
    extracted = extract_audio_from_video(video_path)
    if extracted:
        gr.Info(L("msg_audio_extracted"))
        return gr.update(value=extracted), video_path
    else:
        gr.Info(L("msg_no_audio_track"))
        return gr.update(), video_path


def on_video_clear():
    return ""


def generate_video(
    video_path,
    prompt_audio, text,
    emo_control_method, emo_ref_path, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random,
    max_text_tokens_per_segment,
    *args,
    progress=gr.Progress(),
):
    print(f"[generate_video] video_path = {video_path!r}")
    print(f"[generate_video] prompt_audio = {prompt_audio!r}")
    print(f"[generate_video] text length = {len(text) if text else 0}")

    if not video_path or (isinstance(video_path, str) and not os.path.isfile(video_path)):
        gr.Warning(L("msg_upload_video"))
        return gr.update(visible=False), gr.update(value=None), gr.update()

    if not prompt_audio or (isinstance(prompt_audio, str) and not os.path.isfile(prompt_audio)):
        gr.Warning(L("msg_upload_audio"))
        return gr.update(visible=False), gr.update(value=None), gr.update()

    if not text or not text.strip():
        gr.Warning(L("msg_enter_text"))
        return gr.update(visible=False), gr.update(value=None), gr.update()

    progress(0.02, desc=L("msg_step1"))
    tts.gr_progress = progress
    try:
        audio_path = run_tts(
            prompt_audio, text,
            emo_control_method, emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random, max_text_tokens_per_segment,
            *args,
        )
    except Exception as exc:
        gr.Warning(f"Audio generation failed: {exc}")
        import traceback; traceback.print_exc()
        return gr.update(visible=False), gr.update(value=None), gr.update()

    if not audio_path or not os.path.isfile(audio_path):
        gr.Warning("Audio generation produced no output.")
        return gr.update(visible=False), gr.update(value=None), gr.update()

    gr.Info(L("msg_audio_done"))

    progress(0.40, desc=L("msg_step2_submit"))
    submit = submit_video_synthesis(audio_path=audio_path, video_path=video_path)
    if not submit["success"]:
        gr.Warning(f"DuiX submit failed: {submit['message']}")
        return gr.update(value=audio_path, visible=False), gr.update(value=None), gr.update()

    task_code = submit["task_code"]

    def _poll_progress(pval, msg):
        progress(0.45 + 0.50 * pval, desc=f"{L('msg_step2_duix')}{msg}")

    progress(0.45, desc=L("msg_step2_wait"))
    result = poll_video_result(task_code, progress_callback=_poll_progress)

    if result["success"]:
        host_video = result["video_path"]
        if not os.path.isfile(host_video):
            gr.Warning(f"DuiX reported success but file not found: {host_video}")
            return gr.update(value=audio_path, visible=False), gr.update(value=None), gr.update()
        progress(1.0, desc=L("msg_done"))
        gr.Info(L("msg_video_done"))

        avatar_update = auto_save_avatar(video_path, prompt_audio)

        return (
            gr.update(value=audio_path, visible=False),
            gr.update(value=host_video, visible=True),
            avatar_update,
        )
    else:
        gr.Warning(f"Video synthesis failed: {result['message']}")
        return gr.update(value=audio_path, visible=False), gr.update(value=None), gr.update()


# =========================================================================
# UI Layout
# =========================================================================

VIDEO_CSS = """
#video_upload video {
    max-height: 280px !important;
    width: 100% !important;
    object-fit: contain !important;
    background: #000;
    border-radius: 8px;
}
#output_video video {
    max-height: 400px !important;
    width: 100% !important;
    object-fit: contain !important;
    background: #000;
    border-radius: 8px;
}
#video_upload .upload-container,
#video_upload .wrap {
    max-height: 320px !important;
}
#output_video .wrap {
    max-height: 440px !important;
}
#generate_btn {
    min-height: 50px !important;
    font-size: 1.1em !important;
    background: #2563eb !important;
    border-color: #2563eb !important;
    color: #fff !important;
}
#generate_btn:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
}
#lang_btn {
    min-width: 100px !important;
}
"""

with gr.Blocks(title="AI Editing Integrated Platform", css=VIDEO_CSS) as demo:
    mutex = threading.Lock()
    lang_state = gr.State(value="zh")

    # ── Header ───────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=6):
            gr.HTML('''
            <h2 style="text-align:center; margin-bottom:0.5em; font-weight:800;
                       letter-spacing:0.02em">
                AI Editing Integrated Platform
            </h2>
            ''')
        with gr.Column(scale=1, min_width=120):
            lang_btn = gr.Button(
                UI_LABELS["zh"]["lang_toggle"],
                elem_id="lang_btn",
                size="sm",
                variant="secondary",
            )

    with gr.Tab(UI_LABELS["zh"]["tab_generate"]) as main_tab:

        # ── Main 3-column layout ─────────────────────────────────────────
        with gr.Row(equal_height=False):

            # LEFT
            with gr.Column(scale=1):
                video_upload = gr.Video(
                    label=UI_LABELS["zh"]["source_video"],
                    sources=["upload"],
                    key="video_upload",
                    elem_id="video_upload",
                    height=280,
                )
                uploaded_video_path = gr.State(value="")
                prompt_audio = gr.Audio(
                    label=UI_LABELS["zh"]["ref_audio"],
                    key="prompt_audio",
                    sources=["upload", "microphone"],
                    type="filepath",
                )

            # CENTER
            with gr.Column(scale=1):
                input_text_single = gr.TextArea(
                    label=UI_LABELS["zh"]["text_to_speak"],
                    key="input_text_single",
                    placeholder=UI_LABELS["zh"]["text_placeholder"],
                    info=f"模型版本: {tts.model_version or '1.0'}",
                    lines=6,
                )
                with gr.Row():
                    prompt_idea_input = gr.Textbox(
                        label=UI_LABELS["zh"]["ai_script_idea"],
                        placeholder=UI_LABELS["zh"]["ai_placeholder"],
                        scale=4, lines=1,
                    )
                    prompt_gen_btn = gr.Button(
                        UI_LABELS["zh"]["btn_ai_generate"],
                        scale=1, min_width=80, variant="secondary",
                    )
                generate_btn = gr.Button(
                    UI_LABELS["zh"]["btn_generate_video"],
                    key="generate_btn",
                    elem_id="generate_btn",
                    variant="primary",
                    size="lg",
                )

            # RIGHT
            with gr.Column(scale=1):
                output_video = gr.Video(
                    label=UI_LABELS["zh"]["result_video"],
                    key="output_video",
                    elem_id="output_video",
                    visible=True, height=400,
                )
                output_audio = gr.Audio(
                    label=UI_LABELS["zh"]["generated_audio"],
                    visible=False, key="output_audio",
                )

        # ── Settings ─────────────────────────────────────────────────────
        with gr.Row():
            experimental_checkbox = gr.Checkbox(
                label=UI_LABELS["zh"]["show_experimental"], value=False)
            glossary_checkbox = gr.Checkbox(
                label=UI_LABELS["zh"]["glossary_toggle"],
                value=tts.normalizer.enable_glossary,
            )

        with gr.Accordion(UI_LABELS["zh"]["feature_settings"],
                          open=False) as feature_accordion:
            with gr.Row():
                emo_choices_zh = [UI_LABELS["zh"]["emo_same_as_ref"],
                                  UI_LABELS["zh"]["emo_use_ref_audio"],
                                  UI_LABELS["zh"]["emo_use_vector"]]
                emo_control_method = gr.Radio(
                    choices=emo_choices_zh, type="index",
                    value=emo_choices_zh[0],
                    label=UI_LABELS["zh"]["emo_control_method"])

            with gr.Group(visible=False) as emotion_reference_group:
                with gr.Row():
                    emo_upload = gr.Audio(
                        label=UI_LABELS["zh"]["emo_ref_upload"], type="filepath")

            with gr.Row(visible=False) as emotion_randomize_group:
                emo_random = gr.Checkbox(
                    label=UI_LABELS["zh"]["emo_random_sample"], value=False)

            with gr.Group(visible=False) as emotion_vector_group:
                with gr.Row():
                    with gr.Column():
                        vec1 = gr.Slider(label=UI_LABELS["zh"]["vec_joy"],
                                         minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec2 = gr.Slider(label=UI_LABELS["zh"]["vec_anger"],
                                         minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec3 = gr.Slider(label=UI_LABELS["zh"]["vec_sadness"],
                                         minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec4 = gr.Slider(label=UI_LABELS["zh"]["vec_fear"],
                                         minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    with gr.Column():
                        vec5 = gr.Slider(label=UI_LABELS["zh"]["vec_disgust"],
                                         minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec6 = gr.Slider(label=UI_LABELS["zh"]["vec_depressed"],
                                         minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec7 = gr.Slider(label=UI_LABELS["zh"]["vec_surprise"],
                                         minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        vec8 = gr.Slider(label=UI_LABELS["zh"]["vec_calm"],
                                         minimum=0.0, maximum=1.0, value=0.0, step=0.05)

            with gr.Group(visible=False) as emo_text_group:
                emo_warning_html = gr.HTML(
                    f'<div style="padding:0.5em 0.8em;border-radius:0.5em;'
                    f'background:#ffa87d;color:#000;font-weight:bold">'
                    f'{html.escape(UI_LABELS["zh"]["experimental_warning"])}</div>'
                )
                with gr.Row():
                    emo_text = gr.Textbox(
                        label=UI_LABELS["zh"]["emo_text_label"],
                        placeholder=UI_LABELS["zh"]["emo_text_placeholder"],
                        value="",
                        info=UI_LABELS["zh"]["emo_text_info"])

            with gr.Row(visible=False) as emo_weight_group:
                emo_weight = gr.Slider(
                    label=UI_LABELS["zh"]["emo_weight"],
                    minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        with gr.Accordion(UI_LABELS["zh"]["glossary_title"], open=False,
                          visible=tts.normalizer.enable_glossary) as glossary_accordion:
            glossary_desc_md = gr.Markdown(UI_LABELS["zh"]["glossary_desc"])
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(
                        label=UI_LABELS["zh"]["glossary_term"],
                        placeholder="IndexTTS2")
                    glossary_reading_zh = gr.Textbox(
                        label=UI_LABELS["zh"]["glossary_zh_reading"],
                        placeholder="Index T-T-S 二")
                    glossary_reading_en = gr.Textbox(
                        label=UI_LABELS["zh"]["glossary_en_reading"],
                        placeholder="Index T-T-S two")
                    btn_add_term = gr.Button(
                        UI_LABELS["zh"]["btn_add_term"], scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(value=format_glossary_markdown())

        with gr.Accordion(UI_LABELS["zh"]["advanced_settings"],
                          open=False) as advanced_accordion:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        f"**{UI_LABELS['zh']['gpt2_sampling']}** "
                        f"_{UI_LABELS['zh']['gpt2_sampling_desc']} "
                        f"[Generation strategies]"
                        f"(https://huggingface.co/docs/transformers/main/en/generation_strategies)._"
                    )
                    with gr.Row():
                        do_sample = gr.Checkbox(
                            label="do_sample", value=True,
                            info=UI_LABELS["zh"]["do_sample_info"])
                        temperature = gr.Slider(
                            label="temperature",
                            minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(
                            label="top_p",
                            minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(
                            label="top_k",
                            minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(
                            label="num_beams",
                            value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(
                            label="repetition_penalty", precision=None,
                            value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(
                            label="length_penalty", precision=None,
                            value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(
                        label="max_mel_tokens", value=1500, minimum=50,
                        maximum=tts.cfg.gpt.max_mel_tokens, step=10,
                        info=UI_LABELS["zh"]["max_mel_info"],
                        key="max_mel_tokens")
                with gr.Column(scale=2):
                    gr.Markdown(
                        f'**{UI_LABELS["zh"]["segment_settings"]}** '
                        f'_{UI_LABELS["zh"]["segment_settings_desc"]}_')
                    with gr.Row():
                        initial_value = max(20, min(
                            tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=UI_LABELS["zh"]["max_segment_tokens"],
                            value=initial_value,
                            minimum=20, maximum=tts.cfg.gpt.max_text_tokens,
                            step=2,
                            key="max_text_tokens_per_segment",
                            info=UI_LABELS["zh"]["max_segment_info"])
                    with gr.Accordion(UI_LABELS["zh"]["preview_segments"],
                                      open=True) as preview_accordion:
                        segments_preview = gr.Dataframe(
                            headers=[UI_LABELS["zh"]["col_index"],
                                     UI_LABELS["zh"]["col_content"],
                                     UI_LABELS["zh"]["col_tokens"]],
                            key="segments_preview", wrap=True)
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            ]

        # ── Avatar Gallery (dropdown-based — reliable) ────────────────────
        with gr.Accordion(UI_LABELS["zh"]["saved_avatars"],
                          open=False) as avatar_accordion:
            with gr.Row():
                avatar_dropdown = gr.Dropdown(
                    label=UI_LABELS["zh"]["avatar_dropdown_label"],
                    choices=_get_avatar_dropdown_choices(),
                    value=None,
                    interactive=True,
                    scale=4,
                )
                btn_refresh_avatars = gr.Button(
                    UI_LABELS["zh"]["btn_refresh_avatars"],
                    size="sm", min_width=40, scale=0)
            with gr.Row():
                btn_load_avatar = gr.Button(
                    UI_LABELS["zh"]["btn_load_avatar"],
                    variant="primary", min_width=80)
                btn_delete_avatar = gr.Button(
                    UI_LABELS["zh"]["btn_delete_avatar"],
                    variant="stop", min_width=80)

    # =====================================================================
    # Event bindings
    # =====================================================================

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)
            segments = tts.tokenizer.split_segments(
                text_tokens_list,
                max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = [[i, ''.join(s), len(s)] for i, s in enumerate(segments)]
            return {segments_preview: gr.update(
                value=data, visible=True, type="array")}
        else:
            df = pd.DataFrame(
                [], columns=[L("col_index"), L("col_content"), L("col_tokens")])
            return {segments_preview: gr.update(value=df)}

    def on_add_glossary_term(term, reading_zh, reading_en):
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()
        if not term:
            gr.Warning(L("msg_enter_term"))
            return gr.update()
        if not reading_zh and not reading_en:
            gr.Warning(L("msg_enter_reading"))
            return gr.update()
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        elif reading_en:
            reading = {"en": reading_en}
        else:
            reading = reading_zh or reading_en
        tts.normalizer.term_glossary[term] = reading
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(L("msg_glossary_updated"), duration=1)
        except Exception as e:
            gr.Error(L("msg_glossary_error"))
            print(f"Error: {e}")
            return gr.update()
        return gr.update(value=format_glossary_markdown())

    def on_method_change(emo_control_method):
        vis = lambda *v: tuple(gr.update(visible=x) for x in v)
        if emo_control_method == 1:
            return vis(True, False, False, False, True)
        elif emo_control_method == 2:
            return vis(False, True, True, False, True)
        elif emo_control_method == 3:
            return vis(False, True, False, True, True)
        else:
            return vis(False, False, False, False, False)

    def on_experimental_change(is_experimental, current_mode_index):
        t = UI_LABELS[_current_lang]
        base = [t["emo_same_as_ref"], t["emo_use_ref_audio"],
                t["emo_use_vector"]]
        new_choices = base + [t["emo_use_text"]] if is_experimental else base
        new_index = (current_mode_index
                     if current_mode_index < len(new_choices) else 0)
        return (gr.update(choices=new_choices, value=new_choices[new_index]),)

    def on_glossary_checkbox_change(is_enabled):
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    def on_demo_load():
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(L("msg_glossary_error"))
            print(f"Error: {e}")
        choices = _get_avatar_dropdown_choices()
        return gr.update(value=format_glossary_markdown()), gr.update(choices=choices)

    # ── Wire events ──────────────────────────────────────────────────────

    lang_btn.click(on_lang_toggle,
        inputs=[lang_state],
        outputs=[
            lang_state,              # 0
            video_upload,            # 1
            prompt_audio,            # 2
            input_text_single,       # 3
            prompt_idea_input,       # 4
            prompt_gen_btn,          # 5
            generate_btn,            # 6
            output_video,            # 7
            lang_btn,                # 8
            experimental_checkbox,   # 9
            glossary_checkbox,       # 10
            feature_accordion,       # 11
            emo_control_method,      # 12
            emo_upload,              # 13
            emo_random,              # 14
            emo_weight,              # 15
            emo_text,                # 16
            emo_warning_html,        # 17
            vec1, vec2, vec3, vec4,  # 18-21
            vec5, vec6, vec7, vec8,  # 22-25
            glossary_accordion,      # 26
            glossary_desc_md,        # 27
            glossary_term,           # 28
            glossary_reading_zh,     # 29
            glossary_reading_en,     # 30
            btn_add_term,            # 31
            glossary_table,          # 32
            advanced_accordion,      # 33
            do_sample,               # 34
            max_mel_tokens,          # 35
            max_text_tokens_per_segment,  # 36
            preview_accordion,       # 37
            avatar_accordion,        # 38
            avatar_dropdown,         # 39
            btn_load_avatar,         # 40
            btn_delete_avatar,       # 41
        ])

    prompt_gen_btn.click(on_generate_prompt,
        inputs=[prompt_idea_input],
        outputs=[input_text_single])

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group, emotion_randomize_group,
                 emotion_vector_group, emo_text_group, emo_weight_group])

    experimental_checkbox.change(on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method])

    glossary_checkbox.change(on_glossary_checkbox_change,
        inputs=[glossary_checkbox], outputs=[glossary_accordion])

    input_text_single.change(on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview])

    max_text_tokens_per_segment.change(on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview])

    prompt_audio.upload(update_prompt_audio, inputs=[], outputs=[generate_btn])

    btn_add_term.click(on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table])

    demo.load(on_demo_load, inputs=[],
              outputs=[glossary_table, avatar_dropdown])

    video_upload.change(on_video_upload,
        inputs=[video_upload],
        outputs=[prompt_audio, uploaded_video_path])

    video_upload.clear(on_video_clear,
        inputs=[], outputs=[uploaded_video_path])

    # Avatar gallery events
    btn_refresh_avatars.click(on_avatar_refresh,
        inputs=[],
        outputs=[avatar_dropdown])

    btn_load_avatar.click(on_avatar_load,
        inputs=[avatar_dropdown],
        outputs=[video_upload, prompt_audio, uploaded_video_path])

    btn_delete_avatar.click(on_avatar_delete,
        inputs=[avatar_dropdown],
        outputs=[avatar_dropdown, video_upload, prompt_audio,
                 uploaded_video_path])

    generate_btn.click(
        generate_video,
        inputs=[
            uploaded_video_path,
            prompt_audio, input_text_single,
            emo_control_method, emo_upload, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_segment,
            *advanced_params,
        ],
        outputs=[output_audio, output_video, avatar_dropdown],
    )


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        allowed_paths=[
            _duix.DUIX_HOST_TEMP,
            _duix.AVATARS_DIR,
            os.path.join(current_dir, "avatars"),
            os.path.join(current_dir, "outputs"),
        ],
    )
