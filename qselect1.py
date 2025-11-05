import torch
from transformers import BertTokenizer, BertForSequenceClassification

class QuestionClassifier:
    def __init__(self, roberta_model_path: str):
        """
        初始化问题分类器

        Parameters:
            roberta_model_path: Chinese-RoBERTa-wwm-ext模型路径
        """
        self.roberta_model_path = roberta_model_path
        self.roberta_tokenizer = None
        self.roberta_model = None
        self.label_to_type = {
            0: "factual",      # 事实性
            1: "explanatory",  # 解释性
            2: "inferential",  # 推理性
            3: "boundary",     # 边界测试
            4: "ambiguous",    # 模糊查询
            5: "negative"      # 否定问题
        }
        self.initialize_roberta_model()

    def initialize_roberta_model(self):
        """初始化Chinese-RoBERTa-wwm-ext模型"""
        try:
            print(f"Loading Chinese-RoBERTa-wwm-ext model from {self.roberta_model_path}...")
            self.roberta_tokenizer = BertTokenizer.from_pretrained(self.roberta_model_path)
            self.roberta_model = BertForSequenceClassification.from_pretrained(
                self.roberta_model_path,
                num_labels=6  # 6种问题类型
            )
            
            # 设置为评估模式
            self.roberta_model.eval()
            
            if torch.cuda.is_available():
                self.roberta_model = self.roberta_model.cuda()
                
            print("Chinese-RoBERTa-wwm-ext model loaded successfully!")
            
        except Exception as e:
            print(f"Failed to load RoBERTa model: {str(e)}")
            raise

    def classify_question_type_with_roberta(self, question: str) -> str:
        """使用Chinese-RoBERTa-wwm-ext模型分类问题类型"""
        try:
            # 准备输入
            inputs = self.roberta_tokenizer(
                question, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 前向传播
            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_label = torch.argmax(predictions, dim=1).item()
            
            # 获取预测概率
            probabilities = predictions.cpu().numpy()[0]
            
            # 返回预测类型
            return self.label_to_type.get(predicted_label, "factual"), probabilities
        
        except Exception as e:
            print(f"RoBERTa classification failed: {str(e)}")
            # 失败时回退到规则匹配
            return self.classify_question_type(question), None

    def classify_question_type(self, question: str) -> str:
        """基于规则的问题类型分类（备用方法）"""
        question_lower = question.lower()
        
        # 规则匹配
        if any(word in question_lower for word in ["什么", "哪些", "何时", "谁", "多少", "是否"]):
            if "为什么" in question_lower or "原因" in question_lower:
                return "inferential"
            elif "不" in question_lower or "没" in question_lower or "无" in question_lower:
                return "negative"
            else:
                return "factual"
                
        elif "为什么" in question_lower or "原因" in question_lower or "如何" in question_lower:
            return "inferential"
            
        elif "什么是" in question_lower or "解释" in question_lower:
            return "explanatory"
            
        elif "哪些情况" in question_lower or "什么时候" in question_lower:
            return "boundary"
            
        elif "关于" in question_lower or len(question.strip()) < 6:
            return "ambiguous"
            
        else:
            return "factual"

# 使用示例
if __name__ == "__main__":
    # 配置参数
    roberta_model_path = "/home/dockeruser/lmy/Model/model4_Chinese-RoBERTa-wwm-ext"  
    
    # 初始化分类器
    classifier = QuestionClassifier(roberta_model_path=roberta_model_path)
    
    # 测试问题
    test_questions = [
        "申报产品需要提供哪些关联文件？",
        "什么是生物学评价？",
        "为什么需要进行稳定性研究？",
        "哪些情况下可以免于临床评价？",
        "关于申请材料",
        "不属于监管信息的内容有哪些？"
    ]
    
    # 测试分类
    for question in test_questions:
        question_type, probabilities = classifier.classify_question_type_with_roberta(question)
        print(f"问题: {question}")
        print(f"类型: {question_type}")
        if probabilities is not None:
            print(f"预测概率: {dict(zip(classifier.label_to_type.values(), probabilities))}")
        print()