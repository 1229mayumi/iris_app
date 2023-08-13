# 必要なモジュールのインポート
import joblib
from flask import Flask, request, render_template
from wtforms import Form, FloatField, validators, SubmitField
import numpy as np
import os

# 学習済みモデルをもとに推論する関数
def predict(x):
    # 学習済みモデル(iris.pkl)を読み込み
    model_path = os.path.join(os.getcwd(), 'src/iris.pkl')  # <-- 変更
    model = joblib.load(model_path)  # <-- 変更
    x = x.reshape(1, -1)
    pred_label = model.predict(x)
    return pred_label

# 推論したラベルから花の名前を返す関数
def getName(label):
    if label == 0:
        return "Setosa"
    elif label == 1:
        return "Versiclor"
    elif label == 2:
        return "Virginica"
    else:
        return "Error"
    
# Flaskのインスタンスを作成
app = Flask(__name__)

# 入力フォームの設定を行うクラス
class IrisForm(Form):
    SepalLength = FloatField("がくの長さ（0cm〜10cm)",
                             [validators.InputRequired(),
                              validators.NumberRange(min=0, max=10, message="0〜10の数値を入力してください")]
                             )
    SepalWidth = FloatField("がくの幅（0cm〜5cm)",
                             [validators.InputRequired(),
                              validators.NumberRange(min=0, max=10, message="0〜5の数値を入力してください")]
                             )
    PetalLength = FloatField("花弁の長さ（0cm〜10cm)",
                             [validators.InputRequired(),
                              validators.NumberRange(min=0, max=10, message="0〜10の数値を入力してください")]
                             )
    PetalWidth = FloatField("花弁の幅（0cm〜5cm)",
                             [validators.InputRequired(),
                              validators.NumberRange(min=0, max=10, message="0〜5の数値を入力してください")]
                             )
    # HTML側に表示するsbmitボタンの設定
    submit = SubmitField("判定")
    
# URLにアクセスがあった場合の挙動
# 入力フォームから数値を受け取る → 推論 → 判定結果を結果表示用のHTMLファイル(result.html)に送る
@app.route('/', methods=['GET', 'POST'])
def predicts():
    # フォームの設定 IrisFormクラスをインスタンス化
    irisForm = IrisForm(request.form)
    # POST メソッドの定義
    if request.method == 'POST':
        # 条件に当てはまる場合
        if irisForm.validate() == False:
            return render_template('index.html', forms=irisForm)
        # 条件に当てはまらない場合、推論を実行
        else:
            VarSepalLength = float(request.form["SepalLength"])
            VarSepalWidth = float(request.form["SepalWidth"])
            VarPetalLength = float(request.form["PetalLength"])
            VarPetalWidth = float(request.form["PetalWidth"])
            # 入力された値を ndarray に変換して推論
            x = np.array([VarSepalLength, VarSepalWidth, VarPetalLength, VarPetalWidth])
            pred = predict(x)
            irisName_ = getName(pred)
            return render_template('result.html', irisName=irisName_)
    
    # GETメソッドの定義
    elif request.method == 'GET':
        return render_template('index.html', forms=irisForm)
# アプリケーションの実行
if __name__ == "__main__":
    app.run(debug=False) #デプロイする時にはFalseに変える
