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
- **Chi tiết từng khía cạnh**:
  - stayingpower: 0.7485
  - texture: 0.7848
  - smell: 0.8423
  - price: 0.9844
  - others: 1.0000
  - colour: 0.8507
  - shipping: 0.9061
  - packing: 0.9593

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

#### 3.4.2. Few-shot Learning
**File**: `ViABSA_BP_Few-shot.ipynb`

- **Mô hình**: GPT-4o
- **Phương pháp**: 
  - K-Means clustering để chọn examples đa dạng (70%)
  - Random sampling cho examples khó (30%)
  - Sentence embeddings với all-MiniLM-L6-v2
- **Kết quả**:
  - **F1_ad**: 0.9126
  - **F1_sc**: 0.7729

#### 3.4.3. Chain-of-Thought (CoT)
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
| **Few-shot** | 0.9126 | 0.7729 |
| **Chain-of-Thought** | 0.9060 | 0.7578 |

## 5. Kết luận

- **Machine Learning** cho kết quả tốt nhất về F1_sc (0.8845)
- **Few-shot Learning** cho kết quả tốt nhất về F1_ad (0.9126) trong các phương pháp prompting
- **Chain-of-Thought** không cải thiện đáng kể so với Few-shot thông thường
- Các phương pháp prompting cho kết quả khá tốt mà không cần huấn luyện mô hình

## 6. Cấu trúc thư mục

```
ABSA_Prompting/
├── data/                    # Chứa tập dữ liệu
├── results/                 # Kết quả thực nghiệm
├── mint/                    # Thư viện hỗ trợ
├── ViABSA_BP_EDA.ipynb      # Phân tích dữ liệu
├── ViABSA_BP_ML.ipynb       # Thực nghiệm Machine Learning
├── ViABSA_BP_DL.ipynb       # Thực nghiệm Deep Learning
├── ViABSA_BP_Zero-shot.ipynb # Thực nghiệm Zero-shot
├── ViABSA_BP_Few-shot.ipynb  # Thực nghiệm Few-shot
├── ViABSA_BP_CoT.ipynb       # Thực nghiệm Chain-of-Thought
├── main.py                   # File chính
├── setup.py                  # Cài đặt package
└── README.md                 # Hướng dẫn sử dụng
```

