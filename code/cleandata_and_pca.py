import re
import pandas as pd
import json
import numpy as np
from joblib import dump
from joblib import load
import sklearn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords


def clean_text(text):
    # 使用正则表达式匹配和去除URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # 使用正则表达式去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text):
    # 根据标点符号将文本拆分为句子，保留标点
    sentences = re.split(r'(?<=[。！？])', text)
    # 去除每个句子的前后空格
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def find_common_sentences_in_all_texts(df, column_name):
    # 将所有文本拆分为句子，收集到一个字典中，记录每个句子的出现次数
    sentence_count = {}
    total_text_count = len(df)

    for text in df[column_name]:
        # 使用集合去重，防止同一文本中重复的句子被多次计数
        sentences = set(split_sentences(text))
        for sentence in sentences:
            if sentence in sentence_count:
                sentence_count[sentence] += 1
            else:
                sentence_count[sentence] = 1

    # 找出在每个文本中都重复的句子（即出现次数等于文本总数的句子）
    common_sentences = {sentence for sentence, count in sentence_count.items() if count == total_text_count}

    return common_sentences

def remove_common_sentences(text, common_sentences):
    sentences = split_sentences(text)
    cleaned_sentences = [s for s in sentences if s not in common_sentences]
    return ''.join(cleaned_sentences)

def remove_punctuation_and_lowercase(text):
    # 使用正则表达式移除所有标点符号
    text = re.sub(r'[^\w\s]', '', text)
    # 将文本转换为小写
    text = text.lower()
    return text

def read_and_process(csv_file, output_file):
    """
    读取CSV文件并对Description列进行处理，移除每组中所有文本共有的句子。
    :param csv_file: 输入的CSV文件路径
    :param output_file: 输出的CSV文件路径
    """
    # try:
    # 读取CSV文件
    print("Reading CSV file...")
    data = pd.read_csv(csv_file)
    data = data

        # 检查是否有Description列和Podcast_ID列
    if 'Description' not in data.columns or 'Podcast_ID' not in data.columns:
        print("Error: 'Description' or 'Podcast_ID' column not found in the CSV file.")
        return

    # 移除 NA 值
    print("Removing NA values...")
    data = data.dropna(subset=['Description'])

    cleaned_data = []

    #7lWwL4gxw5jZeWHTMPSIOO 
    for podcast_id, group in data.groupby('Podcast_ID', sort=False):
        # print(podcast_id)

        # group = group["Description"]
        group = group.copy()
        
        cleaned_descriptions = group['Description'].tolist()
        group['Cleaned_Description'] = cleaned_descriptions

        # 首先清理掉 URL 和多余空格
        # print('clean url')
        group['Cleaned_Description'] = group['Cleaned_Description'].apply(clean_text)
        
        if len(group) >= 3:
            # 找出全局重复的句子
            # print('find duplicate_sentences')
            common_sentences = find_common_sentences_in_all_texts(group, 'Cleaned_Description')
            print(common_sentences)
            
            # 删除每个文本中所有重复的句子
            # print('delete duplicate_sentences')
            group['Cleaned_Description'] = group['Cleaned_Description'].apply(lambda text: remove_common_sentences(text, common_sentences))
            # print(group['Description'])

        # 去掉标点符号并将文本转换为小写
        # print('delete punctuation')
        group['Cleaned_Description'] = group['Cleaned_Description'].apply(remove_punctuation_and_lowercase)
        
        # # 移除 NA 值
        # print(group['Podcast_ID'])
        # group = group.dropna(subset=['Cleaned_Description'])
        
        for Cleaned_Description, groups in group.groupby('Cleaned_Description', sort=False):
            first_row = groups.iloc[0].to_dict()
            cleaned_data.append(first_row)

        # cleaned_descriptions.extend(group['Description'].tolist())

    # print(cleaned_descriptions)
    # 更新清理后的描述
    # data['Cleaned_Description'] = cleaned_descriptions

    # 保存处理后的数据到新CSV文件
    # print(cleaned_data.type)
    # print(cleaned_data[0:10])
    
    cleaned_data = pd.DataFrame(cleaned_data)
    
    # 移除 NA 值
    cleaned_data['Cleaned_Description'] = cleaned_data['Cleaned_Description'].astype(str)  # 确保是字符串类型
    cleaned_data['Cleaned_Description'] = cleaned_data['Cleaned_Description'].str.strip()  # 去掉首尾空格
    cleaned_data['Cleaned_Description'] = cleaned_data['Cleaned_Description'].replace(r'^\s*$', None, regex=True) 
    # cleaned_data = cleaned_data[cleaned_data['Cleaned_Description'].str.strip() != '']
    cleaned_data = cleaned_data.dropna(subset=['Cleaned_Description'])
    print(cleaned_data['Cleaned_Description'].isnull().sum())

    
    print("Saving processed data...")
    cleaned_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

    # except Exception as e:
    #     print(f"An error occurred: {e}")
   
def incremental_pca_with_standardization(output_csv, exclude_wordlist, batch_size=3000,  max_features=20000):
    # 读取数据
    data = pd.read_csv(output_csv)
    print('Finish reading data')

    # 初始化 CountVectorizer
    vectorizer = CountVectorizer(stop_words=exclude_wordlist, max_features = max_features)
    X_sparse = vectorizer.fit_transform(data['Cleaned_Description'])
    vocabulary = vectorizer.vocabulary_
    
    # 将特征索引转换为 Python 原生 int
    vocabulary = {key: int(value) for key, value in vocabulary.items()}

    # 保存为 JSON 文件
    with open('vocabulary.json', 'w') as file:
        json.dump(vocabulary, file)

    # 计算全局均值和标准差
    print("Calculating global mean and std...")
    n_samples = X_sparse.shape[0]
    global_mean = np.zeros(X_sparse.shape[1])
    global_std = np.zeros(X_sparse.shape[1])

    # # 遍历批次，计算全局统计信息
    # for i in range(0, n_samples, batch_size):
    #     print(i)
    #     batch = X_sparse[i:i+batch_size].toarray()
    #     global_mean += batch.sum(axis=0)
    #     global_std += (batch ** 2).sum(axis=0)
    
    # 计算全局均值和标准差
    # global_mean /= n_samples
    # global_std = np.sqrt(global_std / n_samples - global_mean ** 2)

    # 初始化 IncrementalPCA
    ipca = IncrementalPCA(n_components=50)

    # 分批次拟合 IncrementalPCA
    print("Fitting IncrementalPCA with standardized data...")
    for i in range(0, n_samples, batch_size * 20):
        print(i)
        batch = X_sparse[i:i+batch_size].toarray()
        # 标准化当前批次
        # batch = (batch - global_mean) / global_std
        ipca.partial_fit(batch)
        
    dump(ipca, 'incremental_pca_model.pkl')
    print("IncrementalPCA model saved to 'incremental_pca_model.pkl'")

    # 获取降维后的结果
    X_pca = []
    for i in range(0, n_samples, batch_size):
        print(i)
        batch = X_sparse[i:i+batch_size].toarray()
        # 标准化当前批次
        # batch = (batch - global_mean) / global_std
        X_pca.append(ipca.transform(batch))
    
    X_pca = np.vstack(X_pca)
    print("Finish Incremental PCA transformation")

    # 打印贡献率
    print("\n每个主成分的贡献率：")
    print(ipca.explained_variance_ratio_)

    # 打印累计贡献率
    print("\n累计贡献率：")
    print(ipca.explained_variance_ratio_.cumsum())
    
    np.save('X_pca.npy', X_pca)

    return X_pca

def use_pca(new_text, vocabulary_path = 'vocabulary.json', pca_model_path = 'incremental_pca_model.pkl'):
    """
    对单个新文本使用预训练的 CountVectorizer 和 PCA 模型进行降维。

    参数：
        new_text (str): 要处理的新文本。
        vocabulary_path (str): 保存的 CountVectorizer 词汇表的 JSON 文件路径。
        pca_model_path (str): 保存的 PCA 模型的文件路径。

    返回：
        numpy.ndarray: 降维后的特征向量。
    """
    data = new_text
    
    if 'Description' not in data.columns or 'Podcast_ID' not in data.columns:
        print("Error: 'Description' or 'Podcast_ID' column not found in the CSV file.")
        return
    
    # 移除 NA 值
    print("Removing NA values...")
    data = data.dropna(subset=['Description'])

    cleaned_data = []

    #7lWwL4gxw5jZeWHTMPSIOO 
    for podcast_id, group in data.groupby('Podcast_ID', sort=False):
        # print(podcast_id)

        # group = group["Description"]
        group = group.copy()
        
        cleaned_descriptions = group['Description'].tolist()
        group['Cleaned_Description'] = cleaned_descriptions

        # 首先清理掉 URL 和多余空格
        # print('clean url')
        group['Cleaned_Description'] = group['Cleaned_Description'].apply(clean_text)
        
        if len(group) >= 3:
            # 找出全局重复的句子
            # print('find duplicate_sentences')
            common_sentences = find_common_sentences_in_all_texts(group, 'Cleaned_Description')
            print(common_sentences)
            
            # 删除每个文本中所有重复的句子
            # print('delete duplicate_sentences')
            group['Cleaned_Description'] = group['Cleaned_Description'].apply(lambda text: remove_common_sentences(text, common_sentences))
            # print(group['Description'])

        group['Cleaned_Description'] = group['Cleaned_Description'].apply(remove_punctuation_and_lowercase)
        
        cleaned_data.append(group)
    
    cleaned_data = pd.concat(cleaned_data, ignore_index=True)
    
    # 移除 NA 值
    cleaned_data['Cleaned_Description'] = cleaned_data['Cleaned_Description'].astype(str)  # 确保是字符串类型
    cleaned_data['Cleaned_Description'] = cleaned_data['Cleaned_Description'].str.strip()  # 去掉首尾空格
    cleaned_data['Cleaned_Description'] = cleaned_data['Cleaned_Description'].replace(r'^\s*$', None, regex=True) 
    # cleaned_data = cleaned_data[cleaned_data['Cleaned_Description'].str.strip() != '']
    cleaned_data = cleaned_data.dropna(subset=['Cleaned_Description'])
    # print(cleaned_data['Cleaned_Description'].isnull().sum())
    
    text = cleaned_data['Cleaned_Description'].tolist()
    
    # 加载词汇表
    with open(vocabulary_path, 'r') as file:
        loaded_vocabulary = json.load(file)
        
    

    # 初始化 CountVectorizer
    vectorizer = CountVectorizer(vocabulary=loaded_vocabulary)

    # 加载 PCA 模型
    print("IncrementalPCA model loaded from 'incremental_pca_model.pkl'")
    pca = load(pca_model_path)

    # 向量化新文本
    X_new_sparse = vectorizer.transform(text)  # 注意要传递一个列表
    X_new_dense = X_new_sparse.toarray()

    # 使用 PCA 模型降维
    X_new_pca = pca.transform(X_new_dense)

    return X_new_pca
    
def tokenize(output_csv, exclude_wordlist):
    data = pd.read_csv(output_csv)
    print('finish read data')
    # 词频向量化，排除指定的词
    vectorizer = CountVectorizer(stop_words=exclude_wordlist)
    X_sparse = vectorizer.fit_transform(data['Cleaned_Description'])
    print('finish create X_sparse')
    scaler = StandardScaler(with_mean=False)  # 使用 with_mean=False 以保持稀疏性
    X_sparse = scaler.fit_transform(X_sparse)
    vocabulary = vectorizer.vocabulary_
    
    with open('vocabulary.json', 'w') as file:
        json.dump(vocabulary, file)
    print('save vocabulary')
    # 使用 TruncatedSVD 进行降维
    svd = TruncatedSVD(n_components=50)
    X_reduced = svd.fit_transform(X_sparse)
    dump(svd, 'svd_model.joblib')
    np.save('X_reduced.npy', X_reduced)
    print('save X_reduced')
    
    return X_reduced, vocabulary

def clean_word(word):
    """移除符号并转换为小写"""
    return re.sub(r'[^\w\s]', '', word).lower()

def get_stopwords(exclude_wordlist):
    """
    获取处理后的停用词列表，并扩展自定义停用词。
    
    参数:
        exclude_wordlist (list): 自定义要排除的停用词列表。
    
    返回:
        list: 清理后的停用词列表。
    """
    # 获取 NLTK 的停用词
    english_stopwords = set(stopwords.words('english'))
    
    # 清理停用词（去符号，转换小写）
    cleaned_stopwords = {clean_word(word) for word in english_stopwords}
    
    # 处理排除的自定义词列表
    cleaned_exclude = {clean_word(word) for word in exclude_wordlist}
    
    # 合并停用词和自定义排除词
    all_stopwords = cleaned_stopwords.union(cleaned_exclude)
    all_stopwords
    
    return list(all_stopwords)

def process_new_episode(new_text, vocabulary_file = 'vocabulary.json', svd_file = 'svd_model.joblib'):
    """
    使用已保存的词典和降维模型对新文本(输入为description)进行处理。
    
    :param new_text: 新的文本, 字符串或字符串列表。输入为description
    :param vocabulary_file: 保存的词典文件路径
    :param svd_file: 保存的 SVD 模型文件路径
    :return: 新文本的降维向量
    """
    new_text = clean_text(new_text)
    
    
    # 加载词典
    with open(vocabulary_file, 'r') as file:
        vocabulary = json.load(file)

    # 初始化 CountVectorizer 并设置词典
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    
    # 将新文本转换为词频矩阵
    if isinstance(new_text, str):
        new_text = [new_text]  # 转为列表形式
    X_new_sparse = vectorizer.transform(new_text)

    # 加载降维模型 (已保存的 TruncatedSVD)
    svd = np.load(svd_file, allow_pickle=True).item()  # 假设 SVD 使用 np.save 保存
    X_new_reduced = svd.transform(X_new_sparse)
    
    return X_new_reduced
    
    

if __name__ == "__main__":
    # 输入和输出文件路径
    input_csv = "podcast_episodes.csv"
    output_csv = "podcast_episodes_processed.csv"
    output_vocabulary = "vocabulary"
    
    exclude_wordlist = ['to', 'our', 'your', 'you', 'i', 'we', "a", "sponsors", "to", "by", "at", "this", "for", "they", "he"
                        , "she", "it", "with", "out", "the", "about", "more", "also", "lot"]
    
    # 获取英语停用词列表
    stop_words = get_stopwords(exclude_wordlist)
    
    # read_and_process(input_csv, output_csv)
    # X, vocabulary = tokenize(output_csv, stop_words)
    X = incremental_pca_with_standardization(output_csv, stop_words)
    # print(X)
    # print(vocabulary)
    
    print("Reading CSV file...")
    data = pd.read_csv(input_csv)
    data = data[0:10]
    X = use_pca(data)
    print(X)
