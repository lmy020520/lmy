import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import Dict, List, Tuple
import pandas as pd

class MedicalQuestionClassifier:
    def __init__(self, model_path: str = None):
        """
        医疗器械问题分类器
        
        Parameters:
            model_path: 预训练模型路径
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 根据您的数据定义6种问题类型
        self.type_mapping = {
            0: "factual",        # 事实性问题 - 询问具体内容、要求
            1: "explanatory",    # 解释性问题 - 询问原因、定义
            2: "inferential",    # 推理性问题 - 条件推断、分析判断
            3: "boundary",       # 边界测试问题 - 特殊情况、例外情况
            4: "ambiguous",      # 模糊查询 - 范围查询、多条件
            5: "negative"        # 否定问题 - 是否、有无等否定形式
        }
        
        self.reverse_mapping = {v: k for k, v in self.type_mapping.items()}
        
        # 医疗器械领域关键词
        self.medical_keywords = {
            'factual': ['提供', '包括', '内容', '哪些', '文件', '资料', '要求', '清单'],
            'explanatory': ['为什么', '原因', '目的', '定义', '含义', '理解', '解释'],
            'inferential': ['若', '如果', '如何', '推断', '确定', '判断', '分析', '应该'],
            'boundary': ['是否', '有无', '无需', '免于', '例外', '特殊情况', '不涉及'],
            'ambiguous': ['相关', '涉及', '部分', '哪些情况', '查找', '定位', '关于'],
            'negative': ['是否所有', '是否必须', '是否均需', '否', '不', '没', '无', '未']
        }
        
        # 初始化模型和组件
        self.initialize_models()
        self.initialize_rule_patterns()
        
    def initialize_models(self):
        """初始化各类模型"""
        try:
            # 初始化BERT tokenizer和模型
            print("Loading BERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path if self.model_path else "hfl/chinese-roberta-wwm-ext"
            )
            self.bert_model = AutoModel.from_pretrained(
                self.model_path if self.model_path else "hfl/chinese-roberta-wwm-ext"
            ).to(self.device)
            
            # 初始化分类头
            self.classifier_head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 6)  # 6分类
            ).to(self.device)
            
            # 加载训练好的分类头权重（如果有）
            # self.classifier_head.load_state_dict(torch.load('classifier_head.pth'))
            
            self.bert_model.eval()
            self.classifier_head.eval()
            
        except Exception as e:
            print(f"Model initialization warning: {e}")
            self.bert_model = None
            
        # 初始化TF-IDF和传统分类器作为备选
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words=['吗', '呢', '吧', '啊', '的', '了', '是']
        )
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_tfidf_trained = False
        
    def initialize_rule_patterns(self):
        """初始化规则匹配模式"""
        self.rule_patterns = {
            'factual': [
                # 事实性查询模式
                r'(提供|包括|包含|哪些|什么).*(文件|资料|内容|要求|信息)',
                r'.*(清单|目录|明细|项目).*',
                r'.*需要.*(什么|哪些)',
                r'.*应.*(提供|包括)'
            ],
            'explanatory': [
                # 解释性查询模式
                r'为什么.*需要|原因.*是|目的.*是',
                r'如何理解|什么是|定义.*是|含义.*是',
                r'为什么.*要|为何.*要',
                r'.*的.*作用|.*的.*意义'
            ],
            'inferential': [
                # 推理性查询模式
                r'若.*如何|如果.*怎样|假如.*应该',
                r'如何.*推断|如何.*确定|如何.*判断',
                r'.*情况下.*需要|.*条件.*下.*应',
                r'是否.*需要|是否.*必须'
            ],
            'boundary': [
                # 边界测试模式
                r'.*是否.*需要|.*有无.*必要',
                r'.*无需.*|.*免于.*|.*例外.*',
                r'.*不.*需要|.*没.*必要',
                r'哪些情况.*可以|什么时候.*不用'
            ],
            'ambiguous': [
                # 模糊查询模式
                r'.*相关.*要求|.*涉及.*内容',
                r'.*哪些部分|.*什么内容',
                r'如何.*查找|如何.*定位',
                r'关于.*的.*信息'
            ],
            'negative': [
                # 否定问题模式
                r'是否所有.*都|是否必须.*',
                r'.*否.*需要|.*不.*必须',
                r'.*没.*要求|.*无.*必要',
                r'是否.*均需|是否.*都要'
            ]
        }
        
    def extract_advanced_features(self, text: str) -> List[float]:
        """提取高级文本特征"""
        features = []
        
        # 1. 长度特征
        features.append(len(text))
        features.append(len(text.strip()))
        
        # 2. 疑问词特征
        question_words = {
            'factual': ['什么', '哪些', '何时', '谁', '多少', '哪'],
            'explanatory': ['为什么', '为何', '原因', '目的', '定义'],
            'inferential': ['若', '如果', '如何', '推断', '应该'],
            'boundary': ['是否', '有无', '哪些情况', '什么时候'],
            'ambiguous': ['相关', '涉及', '部分', '关于'],
            'negative': ['是否所有', '是否必须', '否', '不', '没']
        }
        
        for category, words in question_words.items():
            count = sum(1 for word in words if word in text)
            features.append(count)
            
        # 3. 标点特征
        features.append(1 if '？' in text or '?' in text else 0)
        features.append(1 if '，' in text else 0)
        
        # 4. 结构特征
        features.append(len(re.findall(r'、', text)))  # 列举项数量
        features.append(len(re.findall(r'（.*?）', text)))  # 括号内容
        
        # 5. 关键词密度
        words = jieba.lcut(text)
        total_words = len(words)
        if total_words > 0:
            for category, keywords in self.medical_keywords.items():
                keyword_count = sum(1 for word in words if word in keywords)
                features.append(keyword_count / total_words)
        else:
            features.extend([0] * len(self.medical_keywords))
            
        return features
    
    def rule_based_classify(self, question: str) -> Tuple[str, float]:
        """基于规则和模式匹配的分类"""
        scores = {category: 0 for category in self.type_mapping.values()}
        
        # 1. 模式匹配得分
        for category, patterns in self.rule_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, question)
                scores[category] += len(matches) * 2
        
        # 2. 关键词匹配得分
        words = jieba.lcut(question)
        for word in words:
            for category, keywords in self.medical_keywords.items():
                if word in keywords:
                    scores[category] += 1
        
        # 3. 结构特征得分
        if '若' in question or '如果' in question:
            scores['inferential'] += 3
        if '是否' in question:
            scores['boundary'] += 2
            scores['negative'] += 1
        if '为什么' in question:
            scores['explanatory'] += 3
        if '哪些' in question and '文件' in question:
            scores['factual'] += 3
        if '相关' in question or '涉及' in question:
            scores['ambiguous'] += 2
        
        # 计算置信度
        max_score = max(scores.values())
        total_score = sum(scores.values())
        
        if total_score > 0:
            confidence = max_score / total_score
        else:
            confidence = 0.5  # 默认置信度
            
        predicted_category = max(scores.items(), key=lambda x: x[1])[0]
        
        return predicted_category, confidence
    
    def bert_classify(self, question: str) -> Tuple[str, float]:
        """使用BERT模型进行分类"""
        if self.bert_model is None:
            return self.rule_based_classify(question)
            
        try:
            # 准备输入
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # 获取BERT嵌入
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                # 分类
                logits = self.classifier_head(cls_embedding)
                probabilities = torch.softmax(logits, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
                
            predicted_category = self.type_mapping.get(predicted_idx, "factual")
            return predicted_category, confidence
            
        except Exception as e:
            print(f"BERT classification failed: {e}")
            return self.rule_based_classify(question)
    
    def train_traditional_model(self, questions: List[str], labels: List[str]):
        """训练传统机器学习模型"""
        try:
            # 提取特征
            tfidf_features = self.tfidf_vectorizer.fit_transform(questions)
            
            manual_features = []
            for question in questions:
                manual_features.append(self.extract_advanced_features(question))
            manual_features = np.array(manual_features)
            
            # 合并特征
            from scipy.sparse import hstack
            combined_features = hstack([tfidf_features, manual_features])
            
            # 转换标签
            label_indices = [self.reverse_mapping[label] for label in labels]
            
            # 训练模型
            self.rf_classifier.fit(combined_features, label_indices)
            self.is_tfidf_trained = True
            
        except Exception as e:
            print(f"Traditional model training failed: {e}")
    
    def traditional_classify(self, question: str) -> Tuple[str, float]:
        """使用传统机器学习模型分类"""
        if not self.is_tfidf_trained:
            return self.rule_based_classify(question)
            
        try:
            # 提取特征
            tfidf_feature = self.tfidf_vectorizer.transform([question])
            manual_feature = np.array([self.extract_advanced_features(question)])
            
            # 合并特征
            from scipy.sparse import hstack
            combined_feature = hstack([tfidf_feature, manual_feature])
            
            # 预测
            prediction = self.rf_classifier.predict(combined_feature)[0]
            probabilities = self.rf_classifier.predict_proba(combined_feature)[0]
            confidence = np.max(probabilities)
            
            predicted_category = self.type_mapping.get(prediction, "factual")
            return predicted_category, confidence
            
        except Exception as e:
            print(f"Traditional classification failed: {e}")
            return self.rule_based_classify(question)
    
    def ensemble_classify(self, question: str) -> Dict:
        """集成分类 - 结合多种方法"""
        # 1. 规则分类
        rule_category, rule_confidence = self.rule_based_classify(question)
        
        # 2. BERT分类
        bert_category, bert_confidence = self.bert_classify(question)
        
        # 3. 传统模型分类
        traditional_category, traditional_confidence = self.traditional_classify(question)
        
        # 投票机制
        votes = [rule_category, bert_category, traditional_category]
        
        # 计算每种类型的得票数
        vote_count = {}
        for vote in votes:
            vote_count[vote] = vote_count.get(vote, 0) + 1
        
        # 选择得票最多的类型
        final_category = max(vote_count.items(), key=lambda x: x[1])[0]
        
        # 计算加权置信度
        weights = {
            'rule': 0.3,
            'bert': 0.4, 
            'traditional': 0.3
        }
        
        final_confidence = (
            rule_confidence * weights['rule'] +
            bert_confidence * weights['bert'] +
            traditional_confidence * weights['traditional']
        )
        
        return {
            'final_category': final_category,
            'confidence': final_confidence,
            'rule_result': (rule_category, rule_confidence),
            'bert_result': (bert_category, bert_confidence),
            'traditional_result': (traditional_category, traditional_confidence),
            'vote_count': vote_count
        }
    
    def classify_question(self, question: str, method: str = 'ensemble') -> Dict:
        """
        分类问题的主函数
        
        Parameters:
            question: 用户问题
            method: 分类方法 ('rule', 'bert', 'traditional', 'ensemble')
        """
        if method == 'rule':
            category, confidence = self.rule_based_classify(question)
            return {'category': category, 'confidence': confidence, 'method': 'rule'}
        elif method == 'bert':
            category, confidence = self.bert_classify(question)
            return {'category': category, 'confidence': confidence, 'method': 'bert'}
        elif method == 'traditional':
            category, confidence = self.traditional_classify(question)
            return {'category': category, 'confidence': confidence, 'method': 'traditional'}
        else:
            return self.ensemble_classify(question)

# 训练数据生成器
class TrainingDataGenerator:
    def __init__(self, classifier: MedicalQuestionClassifier):
        self.classifier = classifier
        
    def generate_from_excel_data(self, excel_file_path: str):
        """从Excel文件生成训练数据"""
        try:
            df = pd.read_excel(excel_file_path)
            questions = []
            labels = []
            
            for _, row in df.iterrows():
                question_type = row['问题类型']
                question_text = row['问题']
                
                if pd.notna(question_type) and pd.notna(question_text):
                    # 映射问题类型
                    type_mapping = {
                        '事实性问题': 'factual',
                        '解释性问题': 'explanatory', 
                        '推理性问题': 'inferential',
                        '边界测试问题': 'boundary',
                        '模糊查询': 'ambiguous',
                        '否定问题': 'negative'
                    }
                    
                    if question_type in type_mapping:
                        questions.append(question_text)
                        labels.append(type_mapping[question_type])
            
            return questions, labels
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            return [], []

# 使用示例
if __name__ == "__main__":
    # 初始化分类器
    classifier = MedicalQuestionClassifier(
        model_path="/home/dockeruser/lmy/Model/model4_Chinese-RoBERTa-wwm-ext"
    )
    
    # 生成训练数据并训练传统模型
    data_generator = TrainingDataGenerator(classifier)
    questions, labels = data_generator.generate_from_excel_data("问答对2.xlsx")
    
    if questions and labels:
        print(f"Loaded {len(questions)} training samples")
        classifier.train_traditional_model(questions, labels)
    
    # 测试问题
    test_questions = [
"是否所有医疗器械均需提交‘辐射安全研究’？",
"是否必须委托第三方机构进行产品检验？",
"是否所有产品均需提供‘使用稳定性’研究？",
"是否所有软件均需进行‘网络安全研究’？",
"是否必须开展动物试验？",
"是否所有申报产品均需提供‘不良事件’历史？",
"是否所有产品均需包含‘禁忌证’说明？",
"是否必须提交‘主文档授权信’？",
"是否所有产品均需进行‘临床评价’？",
"是否所有原材料均需提供来源证明？"

    ]
    
    print("\n=== 问题分类测试 ===")
    for question in test_questions:
        result = classifier.classify_question(question, method='ensemble')
        
        print(f"\n问题: {question}")
        print(f"最终分类: {result['final_category']} (置信度: {result['confidence']:.3f})")
        print(f"规则分类: {result['rule_result'][0]} ({result['rule_result'][1]:.3f})")
        print(f"BERT分类: {result['bert_result'][0]} ({result['bert_result'][1]:.3f})")
        print(f"传统分类: {result['traditional_result'][0]} ({result['traditional_result'][1]:.3f})")
        print(f"投票统计: {result['vote_count']}")
        print("-" * 60)

# 保存和加载模型的工具函数
def save_classifier(classifier: MedicalQuestionClassifier, path: str):
    """保存分类器"""
    torch.save(classifier.classifier_head.state_dict(), f"{path}/classifier_head.pth")
    joblib.dump(classifier.rf_classifier, f"{path}/rf_classifier.pkl")
    joblib.dump(classifier.tfidf_vectorizer, f"{path}/tfidf_vectorizer.pkl")

def load_classifier(classifier: MedicalQuestionClassifier, path: str):
    """加载分类器"""
    classifier.classifier_head.load_state_dict(torch.load(f"{path}/classifier_head.pkl"))
    classifier.rf_classifier = joblib.load(f"{path}/rf_classifier.pkl")
    classifier.tfidf_vectorizer = joblib.load(f"{path}/tfidf_vectorizer.pkl")
    classifier.is_tfidf_trained = True