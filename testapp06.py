""" Streamlitによる退職予測AIシステムの開発
"""

from itertools import chain
from operator import index
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 

# 決定木
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier

# 精度評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# データを分割するライブラリを読み込む
from sklearn.model_selection import train_test_split

# データを水増しするライブラリを読み込む
from imblearn.over_sampling import SMOTE

# ロゴの表示用
from PIL import Image

# ディープコピー
import copy

sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def st_display_table(df: pd.DataFrame):
    """
    Streamlitでデータフレームを表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム

    Returns
    -------
    なし
    """

    # データフレームを表示
    st.subheader('データの確認')
    st.table(df)

    # 参考：Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/


def st_display_graph(df: pd.DataFrame, x_col : str):
    """
    Streamlitでグラフ（ヒストグラム）を表示する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        対象のデータフレーム
    x_col : str
        対象の列名（グラフのx軸）

    Returns
    -------
    なし
    """

    fig, ax = plt.subplots()    # グラフの描画領域を準備
    plt.grid(True)              # 目盛線を表示する

    # グラフ（ヒストグラム）の設定
    sns.countplot(data=df, x=x_col, ax=ax)

    st.pyplot(fig)              # Streamlitでグラフを表示する


def ml_dtree(
    X: pd.DataFrame,
    y: pd.Series,
    depth: int) -> list:
    """
    決定木で学習と予測を行う関数
    
    Parameters
    ----------
    X : pd.DataFrame
        説明変数の列群
    y : pd.Series
        目的変数の列
    depth : int
        決定木の深さ

    Returns
    -------
    list: [学習済みモデル, 予測値, 正解率]
    """

    # 決定木モデルの生成（オプション:木の深さ）
    clf = DecisionTreeClassifier(max_depth=depth)

    # 学習
    clf.fit(X, y)

    # 予測
    pred = clf.predict(X)

    # accuracyで精度評価
    score = accuracy_score(y, pred)

    return [clf, pred, score]


def st_display_dtree(clf, features):
    """
    Streamlitで決定木のツリーを可視化する関数
    
    Parameters
    ----------
    clf : 
        学習済みモデル
    features :
        説明変数の列群

    Returns
    -------
    なし
    """

    # 可視化する決定木の生成
    dot = tree.export_graphviz(clf, 
        out_file=None,  # ファイルは介さずにGraphvizにdot言語データを渡すのでNone
        filled=True,    # Trueにすると、分岐の際にどちらのノードに多く分類されたのか色で示してくれる
        rounded=True,   # Trueにすると、ノードの角を丸く描画する。
    #    feature_names=['あ', 'い', 'う', 'え'], # これを指定しないとチャート上で特徴量の名前が表示されない
        feature_names=features, # これを指定しないとチャート上で説明変数の名前が表示されない
    #    class_names=['setosa' 'versicolor' 'virginica'], # これを指定しないとチャート上で分類名が表示されない
        special_characters=True # 特殊文字を扱えるようにする
        )

    # Streamlitで決定木を表示する
    st.graphviz_chart(dot)


def main():
    """ メインモジュール
    """

    # stのタイトル表示
    st.title("退職予測AI\n（Maschine Learning)")

    # サイドメニューの設定
    activities = ["データ確認", "要約統計量", "グラフ表示", "学習と検証", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'データ確認':

        # ファイルのアップローダー
        uploaded_file = st.sidebar.file_uploader("訓練用データのアップロード", type='csv') 

        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字列の判定
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み
                df = pd.read_csv(uploaded_file, encoding=enc) 

                # データフレームをセッションステートに退避（名称:df）
                st.session_state.df = copy.deepcopy(df)

                # スライダーの表示（表示件数）
                cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

                # テーブルの表示
                st_display_table(df.head(int(cnt)))
        
        elif 'df' in st.session_state: 
            df = copy.deepcopy(st.session_state.df)
            
            cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

            st_display_table(df.head(int(cnt)))

        
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '要約統計量':
        
        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)
            
            # 要約統計量の計算
            count_data = pd.DataFrame(df.loc[ : ,"年齢": ].count(axis=0)).T
            
            mean_data = pd.DataFrame(df.loc[ : ,"年齢": ].mean(axis=0)).T
            
            std_data = pd.DataFrame(df.loc[ : ,"年齢": ].std(axis=0)).T
            
            min_data = pd.DataFrame(df.loc[ : ,"年齢": ].min(axis=0)).T
                        
            first_quartile = pd.DataFrame(df.loc[ : ,"年齢": ].quantile(0.25, axis=0)).T
            
            second_quartile = pd.DataFrame(df.loc[ : ,"年齢": ].quantile(0.5, axis=0)).T
            
            third_quartile = pd.DataFrame(df.loc[ : ,"年齢": ].quantile(0.75, axis=0)).T
            
            max_data = pd.DataFrame(df.loc[ : ,"年齢": ].max(axis=0)).T


            # すべての要約統計量を結合
            df2 = pd.concat(
                [
                    count_data,
                    mean_data, 
                    std_data,
                    min_data,
                    first_quartile,
                    second_quartile,
                    third_quartile,
                    max_data,
                ]
            )
            
            # 行ラベルの修正
            df2.set_axis(["count", "mean", "std", "min", "25%", "50%", "75%", "max"], axis=0, inplace=True)            
            
            # 要約統計量の表示
            st_display_table(df2)

            
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == 'グラフ表示':
        

        # セッションステートにデータフレームがあるかを確認
        if 'df' in st.session_state:
            
            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)
            
            df_graph =  copy.deepcopy(df)
            
            # df_graph = df_graph.loc[(df["月給(ドル)"] > 0) & (df["月給(ドル)"] < 2500) ,"月給(ドル)"] = 0
            
            # df_graph = pd.DataFrame(df[(df["月給(ドル)"] > 0) & (df["月給(ドル)"] < 2500 )]) 
                        
            print(df_graph)
            
                        
            x_axis = st.sidebar.selectbox("グラフのX軸",(df.columns))

        

            # グラフの表示
            st_display_graph(df,x_axis)

            
            
        else:
            st.subheader('訓練用データをアップロードしてください')


    if choice == '学習と検証':

        if 'df' in st.session_state:

            # セッションステートに退避していたデータフレームを復元
            df = copy.deepcopy(st.session_state.df)
            
            # 決定木の深さ
            depth_num = st.sidebar.number_input("決定木の深さ(MAX=3)",min_value=1,max_value=3,value=2)
            
            
            
            # 説明変数と目的変数の設定
            df_x = df.drop("退職", axis=1)   # 退職列以外を説明変数にセット
            df_y = df["退職"]                # 退職列を目的変数にセット
            
            # データの分割
            train_x, valid_x, train_y, valid_y = train_test_split(df_x, df_y, train_size=0.7, stratify=df_y)
            
            
            # 決定木による予測
            clf, train_pred, train_scores = ml_dtree(train_x, train_y, depth_num)
            
            # 訓練データー再現率の計算
            recall_rate = recall_score(train_y, train_pred, pos_label="Yes")
            
            # 訓練データー適合率の計算
            precision_rate = precision_score(train_y, train_pred, pos_label="Yes")
        
            # 決定木のツリーを出力
            st_display_dtree(clf,df.columns[1:])
            
            
            # 訓練データーの予測精度
            accuracy,recall,precision = st.columns(3)
            
            with accuracy:
                
                st.header('Accuracy')
                st.write(train_scores)
                
            with recall:
                st.header('Recall')
                st.write(recall_rate)
                
            with precision:
                st.header('Precision')
                st.write(precision_rate)

        else:
            st.subheader('訓練用データをアップロードしてください')
        
            
    if choice == "About":
        image = Image.open('logo.jpg')
        st.image(image)

        st.markdown("Built by Qikilca")
        st.text("Version 0.1")

        st.markdown("For More Information check out   (https://github.com/Qikilca)")

if __name__ == "__main__":
    main()

