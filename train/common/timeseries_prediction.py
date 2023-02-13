import pandas as pd

#reindex():如果某个索引值当前不存在，就会引入缺失值
def timeseries_prediction(df):

    date_list = df.index.tolist()
    date_list.sort()
    date_start = date_list[0]
    date_end = date_list[-1]
    idx = pd.date_range(date_start, date_end)
    idx = [t.strftime("%Y-%m-%d") for t in idx]
    df = df.reindex(idx, fill_value=0)


    return df