# 必要なモジュールのインポート
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import joblib
import os

# データの取得
iris = load_iris()

# 入力変数と出力変数に切り分け
x = iris.data
t = iris.target

# 学習用データとテスト用データに分割
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

# サポートベクターマシンのインスタンスを作成
model = svm.LinearSVC()

# 学習と推論
model.fit(x_train, t_train)
pred = model.predict(x_test)

# 予測精度を確認
print(classification_report(t_test, pred))

# 学習済みモデルを絶対パスで保存
current_directory = os.getcwd()
model_path = os.path.join(current_directory, "src/iris.pkl")
joblib.dump(model, model_path, compress=True)

