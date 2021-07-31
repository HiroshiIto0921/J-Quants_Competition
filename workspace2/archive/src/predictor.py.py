# -*- coding: utf-8 -*-
import io
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm
import lightgbm as lgb
import matplotlib.pyplot as plt


class ScoringService(object):
    # 訓練期間終了日
    TRAIN_END = "2018-12-31"
    # 評価期間開始日
    VAL_START = "2019-02-01"
    # 評価期間終了日
    VAL_END = "2019-12-01"
    # テスト期間開始日
    TEST_START = "2020-01-01"
    # 目的変数
    TARGET_LABELS = ["label_high_20", "label_low_20"]

    # データをこの変数に読み込む
    dfs = None
    # モデルをこの変数に読み込む
    models = None
    # 対象の銘柄コードをこの変数に読み込む
    codes = None
    
    feature_columns = None

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            # "stock_fin_price": f"{dataset_dir}/stock_fin_price.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            cls.dfs[k] = pd.read_csv(v)
            # DataFrameのindexを設定します。
            if k == "stock_price":
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
            elif k in ["stock_fin", "stock_fin_price", "stock_labels"]:
                cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                    cls.dfs[k].loc[:, "base_date"]
                )
                cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # 予測対象の銘柄コードを取得
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_and_label(cls, dfs, codes, feature, label):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
            codes  (array) : target codes
            feature (pd.DataFrame): features
            label (str) : label column name
        Returns:
            train_X (pd.DataFrame): training data
            train_y (pd.DataFrame): label for train_X
            val_X (pd.DataFrame): validation data
            val_y (pd.DataFrame): label for val_X
            test_X (pd.DataFrame): test data
            test_y (pd.DataFrame): label for test_X
        """
        # 分割データ用の変数を定義
        trains_X, vals_X, tests_X = [], [], []
        trains_y, vals_y, tests_y = [], [], []

        # 銘柄コード毎に特徴量を作成
        for code in tqdm(codes):
            # 特徴量取得
            feats = feature[feature["Local Code"] == code]

            # stock_labelデータを読み込み
            stock_labels = dfs["stock_labels"]
            # 特定の銘柄コードのデータに絞る
            stock_labels = stock_labels[stock_labels["Local Code"] == code]

            # 特定の目的変数に絞る
            labels = stock_labels[label].copy()
            # nanを削除
            labels.dropna(inplace=True)
            
            if feats.shape[0] > 0 and labels.shape[0] > 0:
                # 特徴量と目的変数のインデックスを合わせる
                labels = labels.loc[labels.index.isin(feats.index)]
                feats = feats.loc[feats.index.isin(labels.index)]
                labels.index = feats.index

                # データを分割
                _train_X = feats[: cls.TRAIN_END]
                _val_X = feats[cls.VAL_START : cls.VAL_END]
                _test_X = feats[cls.TEST_START :]

                _train_y = labels[: cls.TRAIN_END]
                _val_y = labels[cls.VAL_START : cls.VAL_END]
                _test_y = labels[cls.TEST_START :]

                # データを配列に格納 (後ほど結合するため)
                trains_X.append(_train_X)
                vals_X.append(_val_X)
                tests_X.append(_test_X)

                trains_y.append(_train_y)
                vals_y.append(_val_y)
                tests_y.append(_test_y)
                
        # 銘柄毎に作成した説明変数データを結合します。
        train_X = pd.concat(trains_X)
        val_X = pd.concat(vals_X)
        test_X = pd.concat(tests_X)
        # 銘柄毎に作成した目的変数データを結合します。
        train_y = pd.concat(trains_y)
        val_y = pd.concat(vals_y)
        test_y = pd.concat(tests_y)

        return train_X, train_y, val_X, val_y, test_X, test_y

    @classmethod
    def get_features_for_predict(cls, dfs, code, start_dt="2016-01-01"):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # stock_finデータを読み込み
        stock_fin = dfs["stock_fin"]

        # 特定の銘柄コードのデータに絞る
        fin_data = stock_fin[stock_fin["Local Code"] == code]
        # 特徴量の作成には過去60営業日のデータを使用しているため、
        # 予測対象日からバッファ含めて土日を除く過去90日遡った時点から特徴量を生成します
        n = 90
        # 特徴量の生成対象期間を指定
        fin_data = fin_data.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        # fin_dataのnp.float64のデータのみを取得
        fin_data = fin_data.select_dtypes(include=["float64"])
        # 欠損値処理
        fin_feats = fin_data.fillna(0)

        # stock_priceデータを読み込む
        price = dfs["stock_price"]
        # 特定の銘柄コードのデータに絞る
        price_data = price[price["Local Code"] == code]
        # 終値のみに絞る
        price_data["datetime"] = pd.to_datetime(price_data["EndOfDayQuote Date"])
        price_data = price_data[["datetime", "Local Code", "EndOfDayQuote ExchangeOfficialClose"]]
        price_data.columns = ["datetime", "Local Code", "EndOfDayQuote ExchangeOfficialClose"]
        
        feats = price_data
        # 特徴量の生成対象期間を指定
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :].copy()

        # 財務データの特徴量とマーケットデータの特徴量のインデックスを合わせる
        feats = feats.loc[feats.index.isin(fin_feats.index)]
        fin_feats = fin_feats.loc[fin_feats.index.isin(feats.index)]

        # データを結合
        feats = pd.concat([feats, fin_feats], axis=1).dropna()

        # 欠損値処理を行います。
        feats = feats.replace([np.inf, -np.inf], 0)

        # 終値の3営業日リターン
        feats["return_1days"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(1)
        # 終値の3営業日リターン
        feats["return_2days"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(2)
        # 終値の5営業日リターン
        feats["return_1week"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(5)
        # 終値の10営業日リターン
        feats["return_2week"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(10)
        # 終値の15営業日リターン
        feats["return_3week"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(15)

        # 終値の20営業日リターン
        feats["return_1month"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(20)
        # 終値の25営業日リターン
        feats["return_25days"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(25)

        # 終値の40営業日リターン
        feats["return_2month"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(40)
        # 終値の60営業日リターン
        feats["return_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(60)

        # 終値の1営業日ボラティリティ
        feats["volatility_1days"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(1).std())
        # 終値の2営業日ボラティリティ
        feats["volatility_2days"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(2).std())
        # 終値の5営業日ボラティリティ
        feats["volatility_1week"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(5).std())
        # 終値の10営業日ボラティリティ
        feats["volatility_2week"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(10).std())
        # 終値の15営業日ボラティリティ
        feats["volatility_3week"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(15).std())
        
        # 終値の20営業日ボラティリティ
        feats["volatility_1month"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(20).std())
        # 終値の25営業日ボラティリティ
        feats["volatility_25days"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(25).std())
        # 終値の40営業日ボラティリティ
        feats["volatility_2month"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(40).std())
        # 終値の60営業日ボラティリティ
        feats["volatility_3month"] = (np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(60).std())

        # 終値と1営業日の単純移動平均線の乖離
        feats["MA_gap_1days"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(1).mean()
        )
        # 終値と2営業日の単純移動平均線の乖離
        feats["MA_gap_2days"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(2).mean()
        )
        # 終値と5営業日の単純移動平均線の乖離
        feats["MA_gap_1week"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(5).mean()
        )
        # 終値と10営業日の単純移動平均線の乖離
        feats["MA_gap_2week"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(10).mean()
        )
        # 終値と15営業日の単純移動平均線の乖離
        feats["MA_gap_3week"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(15).mean()
        )

        # 終値と20営業日の単純移動平均線の乖離
        feats["MA_gap_1month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(20).mean()
        )
        # 終値と25営業日の単純移動平均線の乖離
        feats["MA_gap_25days"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(25).mean()
        )
        # 終値と40営業日の単純移動平均線の乖離
        feats["MA_gap_2month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(40).mean()
        )
        # 終値と60営業日の単純移動平均線の乖離
        feats["MA_gap_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(60).mean()
        )

        # 銘柄コードを設定
        feats["Local Code"] = code

        # stock_listデータを読み込み
        stock_list = dfs["stock_list"].copy()
        stock = stock_list[stock_list["Local Code"] == code]
        
        stock = stock[["Local Code", "17 Sector(Code)", "33 Sector(Code)", 
                       "Size Code (New Index Series)", "IssuedShareEquityQuote IssuedShare"]]
        stock.columns = ["Local Code", "Sector17", "Sector33", "SizeGroup", "IssuedShareEquityQuote IssuedShare"]
        stock.loc[stock["SizeGroup"] == "-", "SizeGroup"] = 99
        stock["SizeGroup"] = stock["SizeGroup"].astype(int)
        feats = pd.merge(feats, stock, how="left", on=["Local Code"])

        
        # 財務指標の作成
        # EPS : １株あたり純利益
        feats["EPS"] = (feats["Result_FinancialStatement NetIncome"]*1000000)/feats["IssuedShareEquityQuote IssuedShare"]
        # Forecast_EPS : １株あたり純利益
        feats["Forecast_EPS"] = (feats["Forecast_FinancialStatement NetIncome"]*1000000)/feats["IssuedShareEquityQuote IssuedShare"]

        # Dividend Payout Ratio : 配当性向
        feats.loc[feats["Result_Dividend AnnualDividendPerShare"] == "", "Result_Dividend AnnualDividendPerShare"] = 0.0
        feats["DPR"] = feats["Result_Dividend AnnualDividendPerShare"] / feats["EPS"]
        # Forecast Dividend Payout Ratio : 配当性向
        feats.loc[feats["Forecast_Dividend AnnualDividendPerShare"] == "", "Forecast_Dividend AnnualDividendPerShare"] = 0.0
        feats["Forecast_DPR"] = feats["Forecast_Dividend AnnualDividendPerShare"] / feats["Forecast_EPS"]

        # ROA : 総資本経常利益率
        feats["ROA"] = feats["Result_FinancialStatement OrdinaryIncome"]/feats["Result_FinancialStatement TotalAssets"]

        # PER : 株価収益率
        feats["PER"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
                    feats["Result_FinancialStatement NetIncome"]*1000000/feats["IssuedShareEquityQuote IssuedShare"]
        )
        feats["Result_FinancialStatement CashFlowsFromOperatingActivities"] = \
                    feats["Result_FinancialStatement CashFlowsFromOperatingActivities"] * 1000000
        feats["PER"][feats["Result_FinancialStatement CashFlowsFromOperatingActivities"] == 0] = 0.0

        # Forecast_PER : 株価収益率
        feats["Forecast_PER"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
                    feats["Forecast_FinancialStatement NetIncome"]*1000000/feats["IssuedShareEquityQuote IssuedShare"]
        )

        # PBR : 株価純資産倍率
        feats["PBR"] = feats["EndOfDayQuote ExchangeOfficialClose"]/  (
                   feats["Result_FinancialStatement NetAssets"]*1000000/feats["IssuedShareEquityQuote IssuedShare"]
        )

        # 自己資本比率
        feats["EquityRatio"] = feats["Result_FinancialStatement NetAssets"]/feats["Result_FinancialStatement TotalAssets"]

        # ROE : 自己資本利益率
        feats["ROE"] = 0.0
        feats.loc[feats["PER"] != 0.0, "ROE"] = feats["PBR"]/feats["PER"]

        # Forecast_ROE : 自己資本利益率
        feats["Forecast_ROE"] = 0.0
        feats.loc[feats["Forecast_PER"] != 0.0, "Forecast_ROE"] = feats["PBR"]/feats["Forecast_PER"]

        # Profit Margin : 売上高純利益率
        feats["ProfitMargin"] = feats["Result_FinancialStatement NetIncome"]/feats["Result_FinancialStatement NetSales"]

        # Forecast Profit Margin : 売上高純利益率
        feats["Forecast_ProfitMargin"] = feats["Forecast_FinancialStatement NetIncome"] / feats["Forecast_FinancialStatement NetSales"]

        # 総資本回転率
        feats["TotalAssetTurnover"] = feats["Result_FinancialStatement NetSales"] / feats["Result_FinancialStatement TotalAssets"]
        # 総資本回転率
        feats["Forecast_TotalAssetTurnover"] = feats["Forecast_FinancialStatement NetSales"] / feats["Result_FinancialStatement TotalAssets"]
        # 財務レバレッジ
        feats["FinancialLeverage"] = feats["Result_FinancialStatement TotalAssets"] / feats["Result_FinancialStatement NetAssets"]

        # 売上高伸び率予想
        feats["NetSales_Change"] = (feats["Forecast_FinancialStatement NetSales"] - feats["Result_FinancialStatement NetSales"]) / feats["Result_FinancialStatement NetSales"]

        # 営業利益伸び率予想
        feats["OperatingIncome_Change"] = (feats["Forecast_FinancialStatement OperatingIncome"] - feats["Result_FinancialStatement OperatingIncome"]) / feats["Result_FinancialStatement OperatingIncome"]
        
        # 経常利益伸び率予想
        feats["OrdinaryIncome_Change"] = (feats["Forecast_FinancialStatement OrdinaryIncome"] - feats["Result_FinancialStatement OrdinaryIncome"]) / feats["Result_FinancialStatement OrdinaryIncome"]

        # 純利益伸び率予想
        feats["NetIncome_Change"] = (feats["Forecast_FinancialStatement NetIncome"] - feats["Result_FinancialStatement NetIncome"]) / feats["Result_FinancialStatement NetIncome"]


        # 欠損値処理
        feats = feats.fillna(0)
        # 元データのカラムを削除
        feats = feats.drop(["EndOfDayQuote ExchangeOfficialClose",
                            "EPS", 
                            "Forecast_EPS",
                            "IssuedShareEquityQuote IssuedShare",
                            "Result_Dividend AnnualDividendPerShare",
                            "Forecast_Dividend AnnualDividendPerShare",
                            "Result_FinancialStatement NetIncome",
                            "Forecast_FinancialStatement NetIncome",
                            "Result_FinancialStatement OrdinaryIncome",
                            "Forecast_FinancialStatement OrdinaryIncome",
                            "Result_FinancialStatement OperatingIncome",
                            "Forecast_FinancialStatement OperatingIncome",
                            "Result_FinancialStatement CashFlowsFromOperatingActivities",
                            "Result_FinancialStatement CashFlowsFromFinancingActivities",
                            "Result_FinancialStatement CashFlowsFromInvestingActivities",
                            "Result_FinancialStatement TotalAssets",
                            "Result_FinancialStatement NetAssets",
                            "Result_FinancialStatement NetSales",
                            "Forecast_FinancialStatement NetSales",
                            #"Local Code",
        ], axis=1)

        # 生成対象日以降の特徴量に絞る
        feats = feats[feats["datetime"] >= pd.Timestamp(start_dt)]
        feats.set_index("datetime", inplace=True)

        # 欠損値処理
        feats = feats.replace([np.inf, -np.inf], 0)
        feats = feats.fillna(0)

        return feats

    @classmethod
    def get_feature_columns(cls, dfs, train_X, column_group="fundamental+technical"):
        # 特徴量グループを定義
        # ファンダメンタル
        fundamental_cols = dfs["stock_fin"].select_dtypes("float64").columns
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_Dividend DividendPayableDate"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_Dividend AnnualDividendPerShare"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Forecast_Dividend AnnualDividendPerShare"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement NetIncome"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Forecast_FinancialStatement NetIncome"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement OrdinaryIncome"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Forecast_FinancialStatement OrdinaryIncome"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement OperatingIncome"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Forecast_FinancialStatement OperatingIncome"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement CashFlowsFromOperatingActivities"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement CashFlowsFromFinancingActivities"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement CashFlowsFromInvestingActivities"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement TotalAssets"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement NetAssets"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Result_FinancialStatement NetSales"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Forecast_FinancialStatement NetSales"]
        fundamental_cols = fundamental_cols[fundamental_cols != "Local Code"]
        # 価格変化率
        returns_cols = [x for x in train_X.columns if "return" in x]
        # テクニカル
        technical_cols = [
            x for x in train_X.columns if (x not in fundamental_cols) and (x != "Local Code")
        ]
        columns = {
            "fundamental_only": fundamental_cols,
            "return_only": returns_cols,
            "technical_only": technical_cols,
            "fundamental+technical": list(fundamental_cols) + list(technical_cols),
        }
        return columns[column_group]

    @classmethod
    def create_model(cls, dfs, codes, label):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            codes (list[int]): A local code for a listed company
            label (str): prediction target label
        Returns:
            RandomForestRegressor
        """
        # 特徴量を取得
        buff = []
        #codes = [4307]
        for code in tqdm(codes):
            buff.append(cls.get_features_for_predict(cls.dfs, code))
        feature = pd.concat(buff)
        # 特徴量と目的変数を一致させて、データを分割
        train_X, train_y, val_X, val_y, _, _ = cls.get_features_and_label(cls.dfs, codes, feature, label)
        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(cls.dfs, train_X)

        # モデル作成
        #model = RandomForestRegressor(random_state=0)
        #model.fit(train_X[feature_columns], train_y)

        num_boost_round = 500
        early_stopping_rounds = 10
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'seed': 1029
        }
        
        dtrain = lgb.Dataset(train_X[feature_columns], train_y)
        dvalid = lgb.Dataset(val_X[feature_columns], val_y)
        
        model = lgb.train(
            params, dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=['train', 'valid'],
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )

        # データ数の増加分だけラウンド数を増やす
        best_iteration = int(model.best_iteration * (len(train_X) + len(val_X)) / len(train_X))
        model = lgb.train(
            params, dtrain,
            num_boost_round=best_iteration
        )

        return model

    @classmethod
    def save_model(cls, model, label, model_path="../model"):
        """
        Args:
            model (RandomForestRegressor): trained model
            label (str): prediction target label
            model_path (str): path to save model
        Returns:
            -
        """
        # tag::save_model_partial[]
        # モデル保存先ディレクトリを作成
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f"my_model_{label}.pkl"), "wb") as f:
            # モデルをpickle形式で保存
            pickle.dump(model, f)
        # end::save_model_partial[]

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            m = os.path.join(model_path, f"my_model_{label}.pkl")
            with open(m, "rb") as f:
                # pickle形式で保存されているモデルを読み込み
                cls.models[label] = pickle.load(f)

        return True

    @classmethod
    def train_and_save_model(
        cls, inputs, labels=None, codes=None, model_path="../model"
    ):
        """Predict method

        Args:
            inputs (str)   : paths to the dataset files
            labels (array) : labels which is used in prediction model
            codes  (array) : target codes
            model_path (str): Path to the trained model directory.
        Returns:
            Dict[pd.DataFrame]: Inference for the given input.
        """
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS
        for label in labels:
            print(label)
            model = cls.create_model(cls.dfs, codes=codes, label=label)
            cls.save_model(model, label, model_path=model_path)

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """

        # データ読み込み
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # 予測対象の銘柄コードと目的変数を設定
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # 特徴量を作成
        buff = []
        print("+ get features for predict")
        for code in tqdm(codes):
            buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        feats = pd.concat(buff)

        # 結果を以下のcsv形式で出力する
        # １列目:datetimeとcodeをつなげたもの(Ex 2016-05-09-1301)
        # ２列目:label_high_20　終値→最高値への変化率
        # ３列目:label_low_20　終値→最安値への変化率
        # headerはなし、B列C列はfloat64

        # 日付と銘柄コードに絞り込み
        df = feats.loc[:, ["Local Code"]].copy()
        # codeを出力形式の１列目と一致させる
        df.loc[:, "Local Code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "Local Code"].astype(str)

        # 出力対象列を定義
        output_columns = ["Local Code"]

        # 特徴量カラムを指定
        feature_columns = cls.get_feature_columns(cls.dfs, feats)

        # 目的変数毎に予測
        print("+ prediction start")
        for label in tqdm(labels):
            print(label)
            # 予測実施
            df[label] = cls.models[label].predict(feats[feature_columns])
            # 出力対象列に追加
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()


    @classmethod
    def plot_importance(cls, labels=None):
        if labels is None:
            labels = cls.TARGET_LABELS

        for label in labels:
            print(label)
            lgb.plot_importance(cls.models[label], importance_type='gain', figsize=(8, 10));

