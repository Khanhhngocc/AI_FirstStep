import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
import re #regex


def my_func(loc):
    result = re.findall("\ [A-Z]{2}$", loc) #tìm tất cả, nếu không có trả về list rỗng []
    if len(result) == 1:
        return result[0][1:] #[1:] --> bỏ dấu cách ở đầu
    else:
        return loc


data = pd.read_excel("final_project.ods", engine="odf", dtype=str) #Sẽ xảy ra lỗi không lowercase được số --> vì thế chuyển hết về str
data["location"] = data["location"].apply(my_func)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y) #Chia đều các phân loại class về 2 tệp data (training, set)
ros = SMOTEN(random_state=42, k_neighbors=2, sampling_strategy={"director_business_unit_leader": 500, "specialist": 500,
                                                   "managing_director_small_medium_company": 500})
print(y_train.value_counts())
print("--------------")
x_train, y_train = ros.fit_resample(x_train, y_train)
print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "title"), #không cần ["title"] như bài trước vì TfidfVectorizer chỉ cần chỉ ra tên cột thôi
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1), min_df=0.01, max_df=0.95), #document frequency
     "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words=["english"], ngram_range=(1, 1)), "industry"), #TfidfVectorizer đã tự động gán cho những từ được điểm cao cho mỗi document
    # dựa trên TF và IDF, điểm TF * IDF càng cao càng tốt
    #Khi mà có nhiều từ nhưng mấy từ đó tần suất xuất hiện ít, mô hình vẫn ít nhiều bị ảnh hưởng bởi những từ xấu hiện ít đó
    #khiến xao nhãng hơn cho những từ có tần suất xuất hiện nhiều (những từ mà có tính phân loại các document cao chứ
    # từ mà document nào cũng có thì TF * IDF = 0 rồi --> nên phải loại chọn lọc (bỏ bớt) các từ xuất hiện ít
    #--> vì thế dùng min_df=0.01, max_df=0.95, từ nào dưới min_df hoặc trên max_df là chim cút hết
    #--> giảm số lượng vocabulary, giảm chiều
    #tuỳ cái dùng nhé, chứ cái title chủ yếu là đặc thù, nếu rút bớt đi thì chả còn gì cả
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selector", SelectPercentile(chi2, percentile=10)) #k là số feature tốt nhất muốn giữ lại
    # SelectKBest(chi2, k=100)) nhỡ precessing xong mà số k mình chọn fix cứng mà mình chọn còn nhiều hơn số feature sau khi precessing thì lại k tốt
    # nên là dùng % percentile thay vì ghi số cụ thể
    ("regressor", RandomForestClassifier()),
])

cls.fit(x_train, y_train)
y_predict = cls.predict(x_test)
print(classification_report(y_test, y_predict))
