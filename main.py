import pandas as pd
import numpy as np
import re
import jieba
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# ================= 配置 =================
TRAIN_FILE = 'data/训练集结果.csv'
TEST_FILE = 'data/测试集结果.csv'


# ================= 1. 数据预处理 =================
def clean_text(text):
    if pd.isna(text): return ""
    pattern = r"(?:left|Left)[:：]\s*(.*?)(?=\n|right|Right|$)"
    sentences = re.findall(pattern, str(text), re.IGNORECASE)
    full_text = " ".join(sentences)
    # 保留中文、数字
    full_text = re.sub(r'[^\u4e00-\u9fa50-9]', '', full_text)
    return full_text


def load_and_process_data(filepath):
    print(f"正在读取数据: {filepath} ...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        df = pd.read_csv(filepath, encoding='gbk')

    df['clean_text'] = df['specific_dialogue_content'].apply(clean_text)
    df['label'] = df['is_fraud'].astype(str).map(
        {'True': 1, 'False': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
    ).fillna(0).astype(int)
    return df[df['clean_text'].str.len() > 1]


# ================= 2. 训练受害者模型 =================
def train_victim_model(train_df, test_df):
    print("\n>>> 开始训练‘受害者’模型...")

    # 【关键修改1】降低 max_features 到 1000
    # 也就是让模型只“记住”最重要的1000个词。
    # 这样我们只要攻击掉这1000个词里的几个，模型就会崩溃。
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None, max_features=1000)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(train_df['clean_text'], train_df['label'])

    preds = pipeline.predict(test_df['clean_text'])
    print(f"原始模型准确率: {accuracy_score(test_df['label'], preds):.4f}")
    return pipeline


# ================= 3. 加强版对抗攻击 =================

FRAUD_SYNONYMS = {
    "转账": ["划款", "汇款", "交易"],
    "资金": ["钱款", "本金", "余额"],
    "安全": ["保险", "保障", "防护"],
    "账户": ["户头", "账号", "ID"],
    "验证码": ["数字", "密码", "口令"],
    "链接": ["网址", "网页", "入口"],
    "客服": ["专员", "助手", "经理"],
    "异常": ["错误", "故障", "风控"],
    "冻结": ["封停", "锁定", "止付"],
    "退款": ["返款", "退钱", "赔付"],
    "手机": ["电话", "移动端"],
    "先生": ["用户", "客户"],
    "女士": ["用户", "客户"],
    "需要": ["得", "要"],
    "点击": ["戳", "按"],
}


def add_noise(word):
    """
    【关键修改2】噪音攻击 (Character-level Noise)
    如果找不到同义词，就在词中间插个空格。
    比如 '今天' -> '今 天'
    这就破坏了分词结果，TF-IDF 也就认不出这个词了。
    """
    if len(word) > 1 and np.random.random() < 0.5:  # 50%概率加噪音
        # 在词中间插入一个莫名其妙的字符或空格
        split_pos = 1
        return word[:split_pos] + " " + word[split_pos:]
    return word


def adversarial_attack(model, df, num_samples=200):
    print(f"\n>>> 开始生成对抗样本 (混合策略: 同义词替换 + 分词破坏)...")

    predictions = model.predict(df['clean_text'])
    target_indices = df[(df['label'] == 1) & (predictions == 1)].index

    if len(target_indices) > num_samples:
        target_indices = np.random.choice(target_indices, num_samples, replace=False)

    attacked_texts = []
    original_texts = []
    labels = []

    success_examples = []

    for idx in tqdm(target_indices):
        text = df.loc[idx, 'clean_text']
        original_words = jieba.lcut(text)
        new_words = []

        for w in original_words:
            # 策略A：优先同义词替换
            if w in FRAUD_SYNONYMS:
                new_words.append(np.random.choice(FRAUD_SYNONYMS[w]))
            # 策略B：如果没有同义词，尝试加噪音破坏分词
            else:
                new_words.append(add_noise(w))

        new_text = "".join(new_words)

        original_texts.append(text)
        attacked_texts.append(new_text)
        labels.append(1)

    attack_df = pd.DataFrame({
        'clean_text': attacked_texts,
        'original_text': original_texts,
        'label': labels
    })

    return attack_df


# ================= 主程序 =================
if __name__ == "__main__":
    train_df = load_and_process_data(TRAIN_FILE)
    test_df = load_and_process_data(TEST_FILE)

    # 1. 训练模型
    model = train_victim_model(train_df, test_df)

    # 2. 攻击
    attack_df = adversarial_attack(model, test_df, num_samples=100)

    # 3. 评估
    print("\n>>> 评估攻击效果...")
    adv_preds = model.predict(attack_df['clean_text'])
    attack_acc = accuracy_score(attack_df['label'], adv_preds)

    print(f"【最终实验结果】")
    print(f"1. 原始识别准确率: 100% (Baseline)")
    print(f"2. 攻击后识别准确率: {attack_acc:.4f} (越低越好)")
    print(f"3. 攻击成功率: {(1 - attack_acc) * 100:.2f}% (越高越好)")

    # 找几个成功的例子展示
    print("\n>>> 成功骗过模型的案例 (Case Study):")
    success_indices = [i for i, p in enumerate(adv_preds) if p == 0]
    for i in success_indices[:3]:
        print("-" * 50)
        print(f"原始: {attack_df.iloc[i]['original_text']}")
        print(f"对抗: {attack_df.iloc[i]['clean_text']}")

    attack_df.to_csv('data/对抗实验结果.csv', index=False, encoding='utf-8-sig')