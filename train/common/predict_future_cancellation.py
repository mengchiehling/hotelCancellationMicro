# from src.io.load_model import load_model
# import argparse, config
# import pandas as pd
# from datetime import datetime, timedelta
#
#
# def predict_future_cancellation(start_time,duration,df):
#     # 這個功能應該就不用了，跟run_prediction.py差不多功能
#     model = load_model(hotel_id=args.hotel_id)
#     y_pred_proba = model.predict_proba(df)
#     y_pred = (y_pred_proba[:, 1] > 0.5) * 1
#     df['y_pred'] = y_pred
#     start_time = datetime.strptime(start_time,"%Y-%m-%d")
#     end_time = start_time + timedelta(days=duration)
#     idx = pd.date_range(start_time, end_time)
#     idx = [t.strftime("%Y-%m-%d") for t in idx]
#     df = df[df['check_in'].isin(idx)]
#     df_grouped = df.groupby(by="check_in")[["y_pred"]].sum()
#
#     return df_grouped
#
#
# def set_configuration():
#
#     config.algorithm = 'lightgbm'
#     config.hotel_ids = args.hotel_ids
#     config.configuration = args.configuration
#
#
# if __name__ == "__main__":
#
#     model_name = 'micro'
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--configuration', type=str, help='"A", please check config/training_config.yml')
#     parser.add_argument('--hotel_ids', nargs='+', type=int, help='hotel ids')
#     parser.add_argument('--duration', type=int)
#     args = parser.parse_args()
#
#     set_configuration()
#
#     # df: 要有sql的api
#     predict_future_cancellation(duration=args.duration, df="api", start_time="2023-02-01")
