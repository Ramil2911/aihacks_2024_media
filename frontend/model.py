import catboost
import pickle
import pandas as pd
import sklearn
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
import architecture
import osmnx as ox
from geopy.geocoders import Nominatim
from shapely import geometry
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from shapely.geometry import Polygon
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import json

model = CatBoostRegressor()

class ModelWrapper:
    def __init__(self, file):
        model.load_model(file)

    def predict(self, input_data):
        ## Предобработка входынх данных
        df = input_data
        df = pd.DataFrame(df).join(pd.json_normalize(df['targetAudience']))

        # Данные о районах Москвы
        geolocator = Nominatim(user_agent="leaders-of-digital")
        def get_lat_lon(city_name):
            location = geolocator.geocode(city_name)
            return location.latitude, location.longitude
        def get_osm_polygons(lat, lon, level):
            gdf = ox.geometries.geometries_from_point((lat, lon), dist=50000, tags={'admin_level': str(level)})
            return gdf
        a, b = get_lat_lon("Moscow")
        data = get_osm_polygons(a, b, 8)
        raions_moskvi = pd.DataFrame(row for idx, row in data.iterrows() if type(row['geometry']) == Polygon)[
            ['geometry', 'name']]

        # Функция для поиска по "квадрату"
        def delta_points(point):
            p1 = geometry.Point(point.x + 0.001, point.y + 0.001)
            p2 = geometry.Point(point.x + 0.001, point.y - 0.001)
            p3 = geometry.Point(point.x - 0.001, point.y + 0.001)
            p4 = geometry.Point(point.x - 0.001, point.y - 0.001)
            return [p1, p2, p3, p4]

        # Функция для определения в каком районе находится точка
        def check_point_in_polygons(point, polygons):
            point = geometry.Point(point)
            outp = []
            pts = delta_points(point)
            for p in pts:
                for ind, polygon in enumerate(polygons['geometry']):
                    if polygon.contains(p) and polygons['name'][ind] not in outp:
                        outp.append(polygons['name'][ind])
            if len(outp) > 0:
                return outp
            return -1
        dists = raions_moskvi['name'].to_list()
        mega_df = pd.DataFrame()
        mega_df['district_name'] = dists
        mega_df['district_count'] = 0
        many_tables = []
        # print(df.columns)

# аааааааааааааааааааааааааа
# данные

        for i in df['points']:
            cords = pd.json_normalize(json.loads(i))
            cords['district'] = cords.apply(
                lambda row: check_point_in_polygons((row['lon'], row['lat']), raions_moskvi), axis=1)
            # cords['num_malls'] =
            # cords['num_shop'] =
            iter_df = mega_df.copy()
            for j in cords['district']:
                if j == -1:
                    continue
                for k in j:
                    iter_df.loc[iter_df['district_name'] == k, 'district_count'] += 1
            many_tables.append(iter_df)
        # print(many_tables[0].columns)
        # print(df['name'][0])

        for i in range(len(many_tables)):
            many_tables[i]['name'] = df['name'][i]
            many_tables[i]['gender'] = df['gender'][i]
            many_tables[i]['ageFrom'] = df['ageFrom'][i]
            many_tables[i]['ageTo'] = df['ageTo'][i]
            many_tables[i]['income'] = df['income'][i]


        ### Создание датасета


        # Заполнить новыми данными!!!!!!!

        down = many_tables.copy()


        # Добавление данных с росстата


        down=list(down)
        for i in range(len(down)):
            down[i] = pd.DataFrame(down[i])

        cat_features = [2, 3, 6]
        num_features = [1, 4, 5] + list(range(7, 21))

        # Загрузка данных
        tru_data = pd.read_csv('my_data_convert.csv')
        ainur_arr = tru_data['0'].values
        ilya_arr = down[0].values
        del_ind = []
        for ind, i in enumerate(ainur_arr):
            if i not in ilya_arr:
                del_ind.append(ind)  # [64, 67, 85, 145]
        delta_data = tru_data.loc[del_ind]
        tru_data.drop(index=del_ind, inplace=True)
        delta_data.index = range(142, 146)
        tru_data.sort_values('0', inplace=True)
        tru_data.drop(columns='0', inplace=True)
        tru_data.index = range(142)

        # Итерируемся по всему датасету и заполняем дополнительными данными
        new_down = []
        #delta_data.rename(columns={'0': 0}, inplace=True)
        for i, data in enumerate(down):
            # print(type(data))
            # print(data.columns)
            data.sort_values('district_name')
            data.drop(index=[132, 136, 137, 138, 139], inplace=True)  # ошибочные индексы районов Москвы
            data.index = range(142)

            new_down2 = pd.concat([data, tru_data], axis=1)
            new_down1 = pd.concat([new_down2, delta_data.rename(columns={'0': 'district_name'})])
            new_down1['district_count'] = new_down1['district_count'].fillna(value=0)
            new_down1.index = range(146)
            new_down1['name'] = new_down1['name'].fillna(value=new_down1['name'][0])
            new_down1['gender'] = new_down1['gender'].fillna(new_down1['gender'][0])
            new_down1['ageFrom'] = new_down1['ageFrom'].fillna(new_down1['ageFrom'][0])
            new_down1['ageTo'] = new_down1['ageTo'].fillna(new_down1['ageTo'][0])
            new_down1['income'] = new_down1['income'].fillna(new_down1['income'][0])

            for i, col in enumerate(new_down1.columns):
                 if (i > 6):
                     new_down1[col] = new_down1[col].fillna(value=new_down1[col].median())
                 new_down.append(new_down1)

        # Обработка данных для получения эмбэддингов
        model_params = dict(
            hidden_size=1024,
            num_subspaces=8,
            embed_dim=128,
            num_heads=8,
            dropout=0.05,
            feedforward_dim=1024,
            emphasis=.75,
            mask_loss_weight=2
        )

        encoder = joblib.load('encoder1.pkl')
        scaler = joblib.load('scaler1.pkl')

        X_cats = list()
        X_nums = list()
        n_cats, n_nums = 0, 0
        # print(new_down1)
        # print(type(new_down[0]))
        # for i in range(len(down)):
        new_down1.columns = range(21)
        X_num = new_down1[num_features].to_numpy()
        X_cat = new_down1[cat_features].to_numpy()
        X_cat = encoder.transform(X_cat)
        X_num = scaler.transform(X_num)
        X_cats.append(X_cat)
        X_nums.append(X_num)
        n_cats = X_cat.shape[1]
        n_nums = X_num.shape[1]

        X_nums = np.array(X_nums)
        X_cats = np.array(X_cats)
        X = np.hstack((X_cats.reshape(X_cats.shape[0], -1), X_nums.reshape(X_nums.shape[0], -1)))

        transformer = architecture.TransformerAutoEncoder(num_inputs=X.shape[1],
          n_cats=n_cats*147,
          n_nums=n_nums*147,
          ).cuda()

        transformer.load_state_dict(torch.load('model_checkpoint.pth')['model'])
        transformer.eval()
        print(X)
        dl = DataLoader(dataset=architecture.SingleDataset(X), batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

        features = []
        with torch.no_grad():
            for x in dl:
                features.append(transformer.feature(x.cuda()).detach().cpu().numpy())
        features = np.vstack(features)

        with torch.no_grad():
            predictions = model.predict(features)
        return predictions.item()