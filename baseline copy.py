#完整版
import gradio as gr
import pandas as pd
import json
import re
import os
import docx
import requests
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path
import zipfile
import shutil

# 配置部分
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = "sk-883d825876464ab6966616a3ae887953"
KNOWLEDGE_BASE_PATH = "/home/dockeruser/lmy/二级医疗器械/uft82.csv"
KNOWLEDGE_BASE_PATH_IVD = "/home/dockeruser/lmy/二级医疗器械/数据2.csv"
LOCAL_MODEL_PATH = "/home/dockeruser/lmy/Model/model1"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REQUIRED_FOLDER_CONTENTS = {
    "1.监管信息": [
        "1.1章节目录.docx",
        "1.2申请表.docx",
        "1.3术语、缩写词列表.docx",
        "1.4产品列表.docx",
        "1.5关联文件.docx",
        "1.6申报前与监管机构的联系情况和沟通记录.docx",
        "1.7符合性声明.docx"
    ],
    # 其他文件夹及文件要求与之前一致
}

class MedicalDeviceClassifier:
    def __init__(self):
        self.md_knowledge_base = None
        self.ivd_knowledge_base = None
        self.embedding_model = None
        self.md_embeddings = None
        self.ivd_embeddings = None
        self.initialize_knowledge_bases()

    def initialize_knowledge_bases(self):
        try:
            # 加载医疗器械知识库
            if not os.path.exists(KNOWLEDGE_BASE_PATH):
                raise FileNotFoundError(f"医疗器械知识库文件不存在: {KNOWLEDGE_BASE_PATH}")
            md_df = pd.read_csv(KNOWLEDGE_BASE_PATH)
            md_required_columns = ['row_id', 'desc', 'intended_use', 'name', 'grade']
            if not all(col in md_df.columns for col in md_required_columns):
                raise ValueError("医疗器械CSV缺少必要列")
            self.md_knowledge_base = md_df.to_dict('records')
            
            # 加载IVD知识库
            if not os.path.exists(KNOWLEDGE_BASE_PATH_IVD):
                raise FileNotFoundError(f"IVD知识库文件不存在: {KNOWLEDGE_BASE_PATH_IVD}")
            ivd_df = pd.read_csv(KNOWLEDGE_BASE_PATH_IVD)
            ivd_required_columns = ['row_id', 'intended_use', 'grade']
            if not all(col in ivd_df.columns for col in ivd_required_columns):
                raise ValueError("IVD CSV缺少必要列")
            self.ivd_knowledge_base = ivd_df.to_dict('records')
            
            # 加载嵌入模型
            print(f"正在从本地加载嵌入模型({LOCAL_MODEL_PATH})，设备: {EMBEDDING_DEVICE}...")
            if not os.path.exists(LOCAL_MODEL_PATH):
                raise FileNotFoundError(f"本地模型路径不存在: {LOCAL_MODEL_PATH}")
            self.embedding_model = SentenceTransformer(
                model_name_or_path=LOCAL_MODEL_PATH,
                device=EMBEDDING_DEVICE
            )
            
            # 为两个知识库生成嵌入向量
            print("正在生成医疗器械知识库嵌入向量...")
            md_corpus = [
                f"{row['name']}。{row['desc']}。用于{row['intended_use']}" 
                for row in self.md_knowledge_base
            ]
            self.md_embeddings = self.embedding_model.encode(
                md_corpus, 
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=32
            )
            
            print("正在生成IVD知识库嵌入向量...")
            ivd_corpus = [
                f"预期用途:{row['intended_use']}" 
                for row in self.ivd_knowledge_base
            ]
            self.ivd_embeddings = self.embedding_model.encode(
                ivd_corpus,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=32
            )
            
            print(f"知识库初始化成功，加载{len(self.md_knowledge_base)}条医疗器械记录和{len(self.ivd_knowledge_base)}条IVD记录")
            
        except Exception as e:
            print(f"知识库初始化失败: {str(e)}")
            raise

    def call_deepseek_api(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """你是一位专业的医疗器械/体外诊断试剂分类专家。请严格遵循以下要求：
1. 必须明确给出分类等级（I类/II类/III类）
2. 必须基于提供的知识库条目进行判断
3. 最终回答必须包含以下JSON结构：
```json
{
  "classification": "I/II/III",
  "confidence": 0.0-1.0,
  "rationale": "分类依据说明",
  "matched_id": 匹配的条目ID,
  "missing_info": ["需要补充的字段"]
}
```"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 800
        }
        
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API请求失败: {str(e)}")
        except KeyError:
            raise RuntimeError("API返回格式异常")

    def parse_document(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            full_text = []
            current_line = ""
            
            for para in doc.paragraphs:
                text = para.text.strip()
                
                if not text:
                    if current_line:
                        full_text.append(current_line)
                        current_line = ""
                    continue
                    
                if text.endswith(('。', '；', '！', '？', '）', '」', '.', ';', '!', '?')):
                    if current_line:
                        full_text.append(current_line + text)
                        current_line = ""
                    else:
                        full_text.append(text)
                else:
                    current_line += text
            
            if current_line:
                full_text.append(current_line)
                
            return "\n".join(full_text)
        except Exception as e:
            raise ValueError(f"文档解析失败: {str(e)}")

    def extract_fields(self, text: str, is_ivd: bool = False) -> dict:
        text = re.sub(r'[:：]\s*', ': ', text)
        
        patterns = {
            "desc": [
                r"产品描述[:：]\s*(.+?)(?=\n|$)",
                r"【产品描述】(.+?)(?=【|$)",
                r"描述[:：]\s*(.+?)(?=\n|$)",
                r"(?<=产品概述[:：]).+?(?=\n|$)"
            ],
            "intended_use": [
                r"预期用途[:：]\s*(.+?)(?=\n|$)",
                r"【预期用途】(.+?)(?=【|$)",
                r"用途[:：]\s*(.+?)(?=\n|$)",
                r"用于(.+?)(?=\n|$)",
                r"适用范围[:：]\s*(.+?)(?=\n|$)"
            ],
            "name": [
                r"品名[:：]\s*(.+?)(?=\n|$)",
                r"【产品名称】(.+?)(?=【|$)",
                r"名称[:：]\s*(.+?)(?=\n|$)",
                r"产品名[:：]\s*(.+?)(?=\n|$)",
                r"注册名称[:：]\s*(.+?)(?=\n|$)"
            ]
        }
        
        result = {}
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    result[field] = match.group(1).strip()
                    break
            else:
                result[field] = ""
        
        # 验证字段完整性
        if is_ivd:
            required_fields = ["intended_use"]
            missing = [k for k in required_fields if not result.get(k)]
            if missing:
                raise ValueError(f"体外诊断试剂分类必须提供预期用途")
        else:
            required_fields = ["desc", "intended_use", "name"]
            missing = [k for k in required_fields if not result.get(k)]
            if missing:
                raise ValueError(f"文档中缺少必要字段: {', '.join(missing)}")
        
        return {k: v.strip() for k, v in result.items()}

    def retrieve_candidates(self, query: dict, is_ivd: bool = False, top_k: int = 3) -> List[Dict]:
        if is_ivd:
            if not self.ivd_knowledge_base or self.ivd_embeddings is None:
                raise RuntimeError("IVD知识库未初始化")
            knowledge_base = self.ivd_knowledge_base
            embeddings = self.ivd_embeddings
            query_text = f"预期用途:{query['intended_use']}"
        else:
            if not self.md_knowledge_base or self.md_embeddings is None:
                raise RuntimeError("医疗器械知识库未初始化")
            knowledge_base = self.md_knowledge_base
            embeddings = self.md_embeddings
            query_text = f"{query['name']}。{query['desc']}。用于{query['intended_use']}"
        
        query_embedding = self.embedding_model.encode(
            query_text, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            row = knowledge_base[idx]
            results.append({
                "row_id": row["row_id"],
                "intended_use": row["intended_use"],
                "grade": row["grade"],
                "similarity": float(score)
            })
        
        return results

    def generate_prompt(self, device_info: dict, candidates: list, is_ivd: bool) -> str:
        candidate_str = "\n".join(
            f"【条目#{c['row_id']}】\n"
            f"预期用途: {c['intended_use']}\n" 
            f"类别: {c['grade']}类\n"
            f"语义相似度: {c['similarity']:.2f}\n"
            for c in candidates
        )
        
        device_type = "体外诊断试剂" if is_ivd else "医疗器械"
        
        return f"""请对以下{device_type}进行分类分析：

===== 待分类设备 =====
预期用途: {device_info['intended_use']}
名称: {device_info.get('name', '未提供')}

===== 知识库候选条目 =====
{candidate_str}

请严格按照要求返回JSON格式结果，必须包含：
1. 明确的分类等级（I/II/III）
2. 置信度（0-1）
3. 详细的分类依据
4. 匹配的条目ID（如无可写-1）
5. 需要补充的信息（如无可写空列表）"""

    def process_document(self, file_path: str, is_ivd: bool = False) -> dict:
        try:
            text = self.parse_document(file_path)
            device_info = self.extract_fields(text, is_ivd)
            
            if is_ivd and not device_info['intended_use']:
                raise ValueError("体外诊断试剂分类必须提供预期用途")
            
            candidates = self.retrieve_candidates(device_info, is_ivd)
            prompt = self.generate_prompt(device_info, candidates, is_ivd)
            api_response = self.call_deepseek_api(prompt)
            
            json_match = re.search(r"```json\n(.+?)\n```", api_response, re.DOTALL)
            if not json_match:
                raise ValueError("未找到有效的JSON响应")
            
            result = json.loads(json_match.group(1))
            result["is_ivd"] = is_ivd
            
            required_fields = ["classification", "confidence", "rationale"]
            if not all(field in result for field in required_fields):
                raise ValueError("API返回缺少必要字段")
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "classification": "未知",
                "confidence": 0,
                "rationale": "处理过程中出现错误",
                "is_ivd": is_ivd
            }

def format_output(result: dict) -> str:
    if "error" in result:
        return f"❌ 处理错误: {result['error']}"
    
    class_map = {
        "I": "Ⅰ类（低风险）",
        "II": "Ⅱ类（中风险）",
        "III": "Ⅲ类（高风险）",
        "unknown": "未知类别"
    }
    
    device_type = "体外诊断试剂" if result.get("is_ivd", False) else "医疗器械"
    classification = class_map.get(result["classification"].upper(), class_map["unknown"])
    confidence = f"{result['confidence']*100:.1f}%" if 'confidence' in result else "未知"
    matched_id = result.get("matched_id", "无")
    rationale = result.get("rationale", "无说明")
    
    output = f"""## 🏥 {device_type}分类结果

**🔍 分类等级**: {classification}  
**📊 置信度**: {confidence}  
**🔗 匹配条目ID**: {matched_id}

### 📝 分类依据
{rationale}
"""
    
    if result.get("missing_info"):
        output += f"\n\n⚠️ **需要补充的信息**: {', '.join(result['missing_info'])}"
    
    output += "\n\n---\n*注：本结果基于AI分析生成，仅供参考。正式分类需以监管部门认定为准。*"
    
    return output

def check_folder_structure(zip_file_path: str) -> str:
    if not os.path.isfile(zip_file_path) or not zip_file_path.lower().endswith('.zip'):
        return "❌ 上传文件不是有效的ZIP压缩文件"
    
    try:
        temp_dir = "temp_unzip"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        missing_folders = []
        missing_files = []
        valid_folders = []
        
        top_level_folders = [d for d in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, d))]
        
        if not top_level_folders:
            return "❌ ZIP文件中不包含任何文件夹"
        
        target_root = os.path.join(temp_dir, top_level_folders[0])
        
        for folder_name, required_files in REQUIRED_FOLDER_CONTENTS.items():
            folder_path = os.path.join(target_root, folder_name)
            
            if not os.path.exists(folder_path):
                missing_folders.append(folder_name)
                continue
            
            existing_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.docx')]
            missing = [f for f in required_files if f not in existing_files]
            
            if missing:
                missing_files.append(f"{folder_name}: {', '.join(missing)}")
            else:
                valid_folders.append(folder_name)
        
        result = "📁 **文件夹完整性检查结果**\n\n"
        
        if missing_folders:
            result += f"❌ **缺失的文件夹**: {', '.join(missing_folders)}\n\n"
        
        if missing_files:
            result += f"❌ **缺失的文件**:\n" + "\n".join([f"  - {item}" for item in missing_files]) + "\n\n"
        
        if valid_folders:
            result += f"✅ **完整的文件夹**: {', '.join(valid_folders)}\n\n"
        
        if not missing_folders and not missing_files:
            result += "🎉 **所有文件夹和文件完整，符合要求！**"
        else:
            result += "⚠️ **文件夹或文件缺失，请补充完整后再提交。**"
        
        shutil.rmtree(temp_dir)
        return result
    
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return f"❌ 检查失败: {str(e)}"

classifier = MedicalDeviceClassifier()

with gr.Blocks(title="医疗器械/IVD智能分类系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""<h1 align="center">🏥 医疗器械/体外诊断试剂分类系统</h1>""")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 上传文档")
            file_input = gr.File(
                label="上传文档",
                file_types=[".docx"],
                type="filepath"
            )
            
            classify_type = gr.Radio(
                label="选择分类类型",
                choices=["医疗器械", "体外诊断试剂"],
                value="医疗器械",
                interactive=True
            )
            
            submit_btn = gr.Button("开始分类", variant="primary")
            
            gr.Markdown("### 📝 文档要求")
            gr.Markdown("""
            - **医疗器械**: 需包含产品描述、预期用途和品名
            - **体外诊断试剂**: 必须包含预期用途（品名可选）
            """)
            
            gr.Markdown("### 📁 上传完整文件夹（ZIP压缩）")
            zip_input = gr.File(
                label="上传包含子文件夹的ZIP压缩文件",
                file_types=[".zip"]
            )
            zip_submit_btn = gr.Button("检查文件夹完整性")
            
            gr.Markdown("### 📁 文件夹及文件要求")
            gr.Markdown("""
            请确保上传的ZIP文件包含以下子文件夹及文件：
            - **1.监管信息**: 
              - 1.1 章节目录.docx
              - 1.2 申请表.docx
              - 1.3 术语、缩写词列表.docx
              - 1.4 产品列表.docx
              - 1.5 关联文件.docx
              - 1.6 申报前与监管机构的联系情况和沟通记录.docx
              - 1.7 符合性声明.docx
            - **2.综述资料**: 
              - 2.1 章节目录.docx
              - 2.2 概述.docx
              - 2.3 产品描述.docx
              - 2.4 适用范围和禁忌证.docx
              - 2.5 申报产品上市历史.docx
              - 2.6 其他需说明的内容.docx
            - **3.非临床资料**: 
              - 3.1 章节目录.docx
              - 3.2 产品风险管理资料.docx
              - 3.3 医疗器械安全和性能基本原则清单.docx
              - 3.4 产品技术要求及检验报告相关附件下载.docx
              - 3.5 研究资料.docx
              - 3.6 非临床文献.docx
              - 3.7 稳定性研究.docx
              - 3.8 其他资料.docx
            - **4.临床评价资料**: 
              - 4.1 章节目录.docx
              - 4.2 临床评价资料.docx
              - 4.3 其他资料.docx
            - **5.产品说明书和标签样稿**: 
              - 5.1 章节目录.docx
              - 5.2 产品说明书.docx
              - 5.3 标签样稿.docx
              - 5.4 其他资料.docx
            - **6.质量管理体系文件**: 
              - 6.1 综述.docx
              - 6.2 章节目录.docx
              - 6.3 生产制造信息.docx
              - 6.4 质量管理体系程序.docx
              - 6.5 管理职责程序.docx
              - 6.6 资源管理程序.docx
              - 6.7 产品实现程序.docx
              - 6.8 质量管理体系的测量、分析和改进程序.docx
              - 6.9 其他质量体系程序信息.docx
              - 6.10 质量管理体系核查文件.docx
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 分类结果")
            output = gr.Markdown(
                label="分析结果",
                value="等待分析...",
                show_copy_button=True
            )
            
            folder_output = gr.Markdown(
                label="文件夹检查结果",
                value="等待检查...",
                show_copy_button=True
            )
    
    submit_btn.click(
        fn=lambda f, t: format_output(classifier.process_document(f, t=="体外诊断试剂")),
        inputs=[file_input, classify_type],
        outputs=output
    )
    
    zip_submit_btn.click(
        fn=check_folder_structure,
        inputs=zip_input,
        outputs=folder_output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        show_error=True,
        share = True
    )