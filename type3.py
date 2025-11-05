import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 关键改动
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import re
from collections import Counter
import jieba  # 原代码用了 jieba，但忘记 import

# -------------------- 分类器 --------------------
class InnovativeQuestionClassifier:
    def __init__(self, roberta_model_path: str):
        self.roberta_model_path = roberta_model_path
        self.initialize_models()

        # 问题类型定义
        self.type_hierarchy = {
            'factual': {'subtypes': ['list_query', 'detail_query', 'requirement_query']},
            'explanatory': {'subtypes': ['concept_explain', 'definition_query', 'principle_explain']},
            'inferential': {'subtypes': ['reason_query', 'comparison_query', 'analysis_query']},
            'boundary': {'subtypes': ['condition_query', 'exception_query', 'scope_query']},
            'ambiguous': {'subtypes': ['search_query', 'related_query', 'vague_query']},
            'negative': {'subtypes': ['exclusion_query', 'prohibition_query', 'negation_query']}
        }

        # 医疗器械领域特征词典
        self.domain_features = self.build_domain_feature_dict()

    # -------------------- 初始化模型 --------------------
    def initialize_models(self):
        # ✅ 1. 自动加载你训练好的 RoBERTa 分类器
        self.tokenizer = AutoTokenizer.from_pretrained(self.roberta_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.roberta_model_path)
        self.model.eval()  # 推理模式

        # 2. 传统模型（此处仅声明，未训练）
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.svm_classifier = LinearSVC()
        self.rf_classifier = RandomForestClassifier(n_estimators=100)

    # -------------------- 其余函数全部保持原样 --------------------
    def build_domain_feature_dict(self):
        return {
            'radiation': ['辐射', '放射', '射线', 'X光', 'CT'],
            'testing': ['检验', '测试', '检测', '验证', '试验'],
            'stability': ['稳定性', '有效期', '货架期', '保存期'],
            'cybersecurity': ['网络安全', '数据安全', '信息安全', '漏洞'],
            'animal_test': ['动物试验', '动物实验', '动物研究', '动物模型'],
            'adverse_event': ['不良事件', '副作用', '并发症', '风险事件'],
            'contraindication': ['禁忌证', '禁忌症', '禁用情况', '不宜使用'],
            'authorization': ['授权信', '授权书', '委托书', '主文档'],
            'clinical': ['临床评价', '临床试验', '临床研究', '临床数据'],
            'material': ['原材料', '物料', '材料来源', '原料证明']
        }

    def extract_linguistic_features(self, question):
        features = {}
        features['question_length'] = len(question)
        features['has_question_mark'] = 1 if '?' in question or '？' in question else 0
        features['has_quotes'] = 1 if '‘' in question or '"' in question else 0
        words = jieba.lcut(question)
        features['word_count'] = len(words)
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        question_words = ['什么', '哪些', '为什么', '如何', '是否', '若', '如果']
        for word in question_words:
            features[f'has_{word}'] = 1 if word in question else 0
        return features

    def extract_domain_features(self, question):
        features = {}
        for feature_name, keywords in self.domain_features.items():
            count = sum(1 for keyword in keywords if keyword in question)
            features[f'domain_{feature_name}'] = count
        return features

    def ensemble_classification(self, question):
        roberta_pred, roberta_probs = self.roberta_predict(question)
        feature_pred, feature_confidence = self.feature_based_predict(question)
        rule_pred = self.rule_based_predict(question)
        final_pred = self.fusion_decision(roberta_pred, feature_pred, rule_pred,
                                          roberta_probs, feature_confidence)
        return final_pred

    # ✅ RoBERTa 预测（已改用正确 tokenizer/model）
    def roberta_predict(self, question):
        inputs = self.tokenizer(question, return_tensors="pt",
                              padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]
            pred = int(np.argmax(probs))
        return pred, probs

    def feature_based_predict(self, question):
        return self.rule_based_predict(question), 0.8

    def rule_based_predict(self, question):
        question_lower = question.lower()
        negation_patterns = [
            (r'是否所有.*', 'negative'),
            (r'.*均需.*', 'negative'),
            (r'.*必须.*', 'negative'),
            (r'.*一律.*', 'negative')
        ]
        for pattern, q_type in negation_patterns:
            if re.search(pattern, question_lower):
                return q_type
        if '哪些' in question_lower or '什么' in question_lower:
            return 'factual'
        elif '为什么' in question_lower or '原因' in question_lower:
            return 'inferential'
        elif '什么是' in question_lower or '解释' in question_lower:
            return 'explanatory'
        elif '若' in question_lower or '如果' in question_lower:
            return 'boundary'
        elif '如何查找' in question_lower or '哪些部分' in question_lower:
            return 'ambiguous'
        return 'factual'

    def fusion_decision(self, roberta_pred, feature_pred, rule_pred, roberta_probs, feature_conf):
        if rule_pred == feature_pred:
            return rule_pred
        if max(roberta_probs) > 0.8:
            return roberta_pred
        return rule_pred

    def hierarchical_classification(self, question):
        main_type = self.ensemble_classification(question)
        subtype = self.classify_subtype(question, main_type)
        return main_type, subtype

    def classify_subtype(self, question, main_type):
        if main_type == 'negative':
            if '所有' in question or '均需' in question:
                return 'exclusion_query'
            elif '必须' in question or '强制' in question:
                return 'prohibition_query'
            else:
                return 'negation_query'
        return 'default_subtype'

    def explain_classification(self, question):
        main_type, subtype = self.hierarchical_classification(question)
        features = self.extract_linguistic_features(question)
        domain_features = self.extract_domain_features(question)
        explanation = {
            'question': question,
            'main_type': main_type,
            'subtype': subtype,
            'key_features': {
                'linguistic': [k for k, v in features.items() if v > 0],
                'domain': [k for k, v in domain_features.items() if v > 0]
            },
            'confidence_factors': self.calculate_confidence_factors(question, main_type)
        }
        return explanation

    def calculate_confidence_factors(self, question, predicted_type):
        factors = {}
        patterns = {
            'negative': [r'是否所有.*', r'.*均需.*', r'.*必须.*'],
            'factual': [r'.*哪些.*', r'.*什么.*'],
            'explanatory': [r'什么是.*', r'.*解释.*']
        }
        match_count = 0
        for pattern in patterns.get(predicted_type, []):
            if re.search(pattern, question):
                match_count += 1
        factors['pattern_match'] = match_count / len(patterns.get(predicted_type, [1]))
        domain_keywords = {
            'negative': ['所有', '均需', '必须', '一律'],
            'factual': ['哪些', '什么', '包含', '包括']
        }
        keyword_count = 0
        for keyword in domain_keywords.get(predicted_type, []):
            if keyword in question:
                keyword_count += 1
        factors['keyword_match'] = keyword_count / len(domain_keywords.get(predicted_type, [1]))
        return factors

# -------------------- 测试 --------------------
if __name__ == "__main__":
    cls = InnovativeQuestionClassifier("/home/lmy/study/nmpa_roberta_cls")
    tests = [
"文件中提到‘风险管理’的部分有哪些？",
"如何查找‘灭菌’相关要求？",
"生物学评价”在哪些情况下可豁免试验？",
"哪些条款涉及‘临床试验’的豁免？",
"如何快速定位‘境外申请人’的文件要求？",
"文件中哪些部分涉及‘稳定性’要求？",
"网络安全”研究需包含哪些子项？",
"如何找到‘质量管理体系核查’的具体文件清单？",
"哪些内容与‘产品包装’直接相关？",
"非临床资料”中哪些研究可能涉及动物试验？"
    ]
    for q in tests:
        exp = cls.explain_classification(q)
        print(f"\n问题: {q}")
        print(f"主类型: {exp['main_type']}")
        print(f"子类型: {exp['subtype']}")
        print(f"关键特征: {exp['key_features']}")
        print(f"置信因子: {exp['confidence_factors']}")