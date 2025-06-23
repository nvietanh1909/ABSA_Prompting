# Khảo sát Kỹ thuật Prompting cho Bài toán phân tích cảm xúc (dựa trên khía cạnh)

Dự án này tập trung vào việc khảo sát các kỹ thuật prompting hiện đại áp dụng cho bài toán phân tích cảm xúc dựa trên khía cạnh (Aspect-based Sentiment Analysis – ABSA). Bên cạnh đó, dự án cũng tiến hành so sánh hiệu quả của các phương pháp cổ điển như **học máy** (machine learning) và **học sâu** (deep learning) với các kỹ thuật prompting mới. Đặc biệt, dự án sẽ áp dụng các kỹ thuật In-context Learning để tăng cường khả năng suy luận chuỗi (Chain-of-Thought – CoT) cho bài toán ABSA, nhằm nâng cao độ chính xác và khả năng giải thích của mô hình.

## 1. Tập dữ liệu
* [ViABSA_BP](https://github.com/linh222/Aspect-based-Sentiment-Analysis-for-Vietnamese-Reviews-about-Beauty-Product-on-E-commerce-Websites) - Aspect-based Sentiment Analysis for Vietnamese Reviews about Beauty Product on {E}-commerce Websites
* [VLSP_2018](https://github.com/ds4v/absa-vlsp-2018) - Multi-task Solution for Aspect Category Sentiment Analysis on Vietnamese Datasets

## 2. Các thư viện sử dụng

* Cài đặt thư viện ```mint``` băng lệnh: ```pip install -e .```
* Jupyter Notebook
* Các thư viện

```
pip install numpy pandas
pip install matplotlib seaborn
pip install -U scikit-learn
pip install python-dotenv
pip install openai
pip install tensorflow
pip install tf-keras
pip install sentence-transformers

```
* Tạo file ```.env``` -> copy các API_KEY vào
* Tạo các folders: ```data```, ```results```
* Tải các tập dữ liệu (mục 1) vào folder ```data```

## 3. Hướng dẫn thực hiện thực nghiệm

### 3.1. Phân tích dữ liệu (EDA)
**File**: `ViABSA_BP_EDA.ipynb`

- **Mục tiêu**: Khám phá và phân tích tập dữ liệu ViABSA_BP
- **Nội dung chính**:
  - Thống kê cơ bản về tập dữ liệu (12,981 mẫu train, 1,623 mẫu test)
  - Phân phối các khía cạnh: stayingpower, texture, smell, price, others, colour, shipping, packing
  - Phân bố cảm xúc theo từng khía cạnh (positive, neutral, negative)
  - Phân tích số lượng khía cạnh trong mỗi đánh giá
  - Giải thích các khái niệm STL (Single-Task Learning) và MTL (Multi-Task Learning)
  - Định nghĩa các độ đo F1_ad (Aspect Detection) và F1_sc (Sentiment Classification)

### 3.2. Các thực nghiệm Machine Learning
**File**: `ViABSA_BP_ML.ipynb`

- **Phương pháp**: TF-IDF + Logistic Regression
- **Kiến trúc**: 
  - Multi-label classification cho Aspect Detection
  - One-vs-Rest classification cho Sentiment Classification
- **Kết quả**:
  - **F1_ad (Aspect Detection)**: 0.9307
  - **F1_sc (Sentiment Classification)**: 0.8845

### 3.3. Các thực nghiệm Deep Learning
**File**: `ViABSA_BP_DL.ipynb`

#### 3.3.1. BiLSTM
- **Kiến trúc**: Embedding + Bidirectional LSTM + Dense
- **Kết quả**:
  - **F1_ad**: 0.8902
  - **F1_sc**: 0.8435

#### 3.3.2. BiGRU
- **Kiến trúc**: Embedding + Bidirectional GRU + Dense
- **Kết quả**:
  - **F1_ad**: 0.8032
  - **F1_sc**: 0.8441

#### 3.3.3. BiLSTM + Conv1D
- **Kiến trúc**: Embedding + Bidirectional LSTM + Conv1D + GlobalMaxPooling + Dense
- **Kết quả**:
  - **F1_ad**: 0.9271
  - **F1_sc**: 0.6906

#### 3.3.4. BiGRU + Conv1D
- **Kiến trúc**: Embedding + Bidirectional GRU + Conv1D + GlobalMaxPooling + Dense
- **Kết quả**: Tương tự BiLSTM + Conv1D

### 3.4. Các thực nghiệm Prompting

#### 3.4.1. Zero-shot Learning
**File**: `ViABSA_BP_Zero-shot.ipynb`

- **Mô hình**: GPT-4o
- **Phương pháp**: Function calling với OpenAI API
- **Kết quả**:
  - **F1_ad**: 0.8700
  - **F1_sc**: 0.7277

#### 3.4.2. Few-shot Learning (Clustering-based)
**File**: `ViABSA_BP_Few-shot-Clustering.ipynb`

- **Mô hình**: GPT-4o
- **Phương pháp**: 
  - Sử dụng Sentence Embedding (all-MiniLM-L6-v2) để biểu diễn các câu.
  - K-Means clustering để chọn ra các example đa dạng nhất (70% số lượng example).
  - Kết hợp chọn ngẫu nhiên các ví dụ "khó" (30%) từ phần còn lại.
  - Prompt cố định cho mọi input.
- **Kết quả**: (Cập nhật theo thực nghiệm của bạn)
  - **F1_ad**: ...
  - **F1_sc**: ...

#### 3.4.3. Few-shot Learning (Similarity-based, Retrieval-based Dynamic Prompting)
**File**: `ViABSA_BP_Few-shot-Similarity.ipynb`

- **Mô hình**: GPT-4o
- **Phương pháp**: 
  - Sử dụng Sentence Embedding (all-MiniLM-L6-v2) để tính độ tương đồng semantic giữa input và các example.
  - Với mỗi input, chọn top-k example gần nhất (retrieval-based, dynamic few-shot).
  - Prompt được xây dựng động cho từng input.
- **Kết quả**:
  - **F1_ad**: 0.8883
  - **F1_sc**: 0.7633

#### 3.4.4. Chain-of-Thought (CoT)
**File**: `ViABSA_BP_CoT.ipynb`

- **Mô hình**: GPT-4o
- **Phương pháp**: 
  - Kết hợp Few-shot learning với Chain-of-Thought reasoning
  - Yêu cầu mô hình suy luận từng bước
- **Kết quả**:
  - **F1_ad**: 0.9060
  - **F1_sc**: 0.7578

## 4. So sánh kết quả

| Phương pháp | F1_ad | F1_sc |
|-------------|-------|-------|
| **Machine Learning** | 0.9307 | 0.8845 |
| **BiLSTM** | 0.8902 | 0.8435 |
| **BiGRU** | 0.8032 | 0.8441 |
| **BiLSTM + Conv1D** | 0.9271 | 0.6906 |
| **Zero-shot** | 0.8700 | 0.7277 |
| **Few-shot (Clustering)** | 0.9126 | 0.7729 |
| **Few-shot (Similarity)** | 0.8883 | 0.7633 |
| **Chain-of-Thought** | 0.9060 | 0.7578 |

## 5. Kết luận

- **Machine Learning** cho kết quả tốt nhất về F1_sc (0.8845)
- **Few-shot Learning** (Clustering hoặc Similarity) cho kết quả tốt về F1_ad trong các phương pháp prompting
- **Chain-of-Thought** không cải thiện đáng kể so với Few-shot thông thường
- Các phương pháp prompting cho kết quả khá tốt mà không cần huấn luyện mô hình

## 6. Cấu trúc thư mục

```
ABSA_Prompting/
├── data/                        # Chứa tập dữ liệu
├── results/                     # Kết quả thực nghiệm
├── mint/                        # Thư viện hỗ trợ
├── ViABSA_BP_EDA.ipynb          # Phân tích dữ liệu
├── ViABSA_BP_ML.ipynb           # Thực nghiệm Machine Learning
├── ViABSA_BP_DL.ipynb           # Thực nghiệm Deep Learning
├── ViABSA_BP_Zero-shot.ipynb    # Thực nghiệm Zero-shot
├── ViABSA_BP_Few-shot-Clustering.ipynb   # Few-shot clustering-based
├── ViABSA_BP_Few-shot-Similarity.ipynb   # Few-shot similarity-based (retrieval)
├── ViABSA_BP_CoT.ipynb          # Thực nghiệm Chain-of-Thought
├── main.py                      # File chính
├── setup.py                     # Cài đặt package
└── README.md                    # Hướng dẫn sử dụng
```

