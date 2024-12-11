import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import geopandas as gpd
from shapely.geometry import Point
import geopy.distance as distance

import planetary_computer as pc
from pystac_client import Client

from datetime import datetime
from datetime import timedelta

import requests
from PIL import Image
from io import BytesIO

import rioxarray
import odc.stac
import tempfile
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import hab_functions

import pygrib

import warnings
warnings.filterwarnings('ignore')

metadata = pd.read_csv('./Data/metadata.csv', parse_dates=['date'])

metadata.head()

metadata.info()

metadata['split'].value_counts(normalize=True)

metadata['date'].value_counts()

train_labels = pd.read_csv('Data/train_labels.csv')

train_labels.head()

train_labels['severity'].value_counts()

train_labels['severity'].value_counts(normalize=True)

train_labels['region'].value_counts()

train_labels['region'].value_counts(normalize=True)

train_regions = train_labels.groupby('region').mean()

for index in train_regions.index:
    train_regions.rename({index:index.title()}, inplace=True)

metadata.agg({'date': ['min', 'max']})

metadata = metadata.set_index('date')

metadata.sort_index(inplace=True)

metadata.head()

seasons = {12: 'winter', 1: 'winter', 2: 'winter', 3:'spring', 4:'spring', 5:'spring',
           6:'summer', 7:'summer', 8:'summer', 9:'autumn', 10:'autumn', 11:'autumn'}

metadata['season'] = metadata.index.month.map(seasons)

metadata.head()

metadata['season'].value_counts(normalize=True)

X_train = metadata[metadata['split'] == 'train'].drop('split', axis=1)

X_test = metadata[metadata['split'] == 'test'].drop('split', axis=1)

X_train = X_train.merge(train_labels, on='uid').set_index(X_train.index)

X_train.head()

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

X_train_geo_df = gpd.GeoDataFrame(X_train, 
                                  geometry=gpd.points_from_xy(X_train['longitude'],
                                                              X_train['latitude']))
X_test_geo_df = gpd.GeoDataFrame(X_test, 
                                 geometry=gpd.points_from_xy(X_test['longitude'], 
                                                             X_test['latitude']))

meta_geo_df = gpd.GeoDataFrame(metadata, 
                               geometry=gpd.points_from_xy(metadata['longitude'], 
                                                           metadata['latitude']))

X_train_geo_df.head()

X_train_months = X_train.groupby(pd.Grouper(freq = 'MS')).count()

train_roll_mean = X_train_months['uid'].rolling(window=12).mean()

X_test_months = X_test.groupby(pd.Grouper(freq = 'MS')).count()

test_roll_mean = X_test_months['uid'].rolling(window=12).mean()

west_df = X_train[X_train['region']=='west']
south_df = X_train[X_train['region']=='south']
northeast_df = X_train[X_train['region']=='northeast']
midwest_df = X_train[X_train['region']=='midwest']

west_severity_monthly = west_df.groupby(pd.Grouper(freq = 'MS')).mean()
west_severity_rolling = west_severity_monthly['severity'].rolling(window=12).mean()

south_severity_monthly = south_df.groupby(pd.Grouper(freq = 'MS')).mean()

south_severity_monthly = south_severity_monthly.interpolate(method='slinear')
south_severity_rolling = south_severity_monthly['severity'].rolling(window=12).mean()

n_e_severity_monthly = northeast_df.groupby(pd.Grouper(freq = 'MS')).mean()

n_e_severity_monthly = n_e_severity_monthly.interpolate(method='linear')
n_e_severity_rolling = n_e_severity_monthly['severity'].rolling(window=12).mean()

midwest_severity_monthly = midwest_df.groupby(pd.Grouper(freq = 'MS')).mean()

midwest_severity_monthly = midwest_severity_monthly.interpolate(method='linear')
midwest_severity_rolling = midwest_severity_monthly['severity'].rolling(window=12).mean()

X_train_annual = X_train['uid'].resample('A').count()
X_test_annual = X_test['uid'].resample('A').count()

annual_df = pd.DataFrame(X_train_annual.values, 
                         index=X_train_annual.index, 
                         columns=['Train'])

annual_df = pd.concat([annual_df, X_test_annual], axis=1)
annual_df.rename(columns={'uid': 'Test'}, inplace=True)
annual_df.set_axis(annual_df.index.year, inplace=True)

sat_df = metadata.reset_index()

sat_df['split'].value_counts()

sat_train = sat_df[sat_df['split'] == 'train'].copy()
sat_test = sat_df[sat_df['split'] == 'test'].copy()

sat_train = sat_train.merge(train_labels, on='uid')

hab_functions.get_important_info(sat_train, dist=31, 
                             big_crop_dist=3000, 
                             small_crop_dist=500, 
                             tiny_crop_dist=100);


sample_row = sat_train[sat_train['uid']=='garm'].iloc[0]

bbox =sample_row['bbox']
date = sample_row['date_range']

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", 
    modifier=pc.sign_inplace
)

search = catalog.search(
    collections=["sentinel-2-l2a", "landsat-c2-l2"],
    bbox=bbox,
    datetime=date,
    query={'eo:cloud_cover': {'lt':100}}
)

search_items = [item for item in search.get_all_items()]

search_items[5].properties

pic_details = []
for pic in search_items:
    pic_details.append(
    {
    'item': pic,
    'satelite_name':pic.collection_id,
    'img_date':pic.datetime.date(),
    'cloud_cover(%)': pic.properties['eo:cloud_cover'],
    'img_bbox': pic.bbox,
    'min_long': pic.bbox[0],
    "max_long": pic.bbox[2],
    "min_lat": pic.bbox[1],
    "max_lat": pic.bbox[3]
    }
    )



big_buffer_df = pd.DataFrame(pic_details)

big_buffer_df['has_sample_point'] = (
        (big_buffer_df.min_lat < sample_row.latitude)
        & (big_buffer_df.max_lat > sample_row.latitude)
        & (big_buffer_df.min_long < sample_row.longitude)
        & (big_buffer_df.max_long > sample_row.longitude)
    )

big_buffer_df = big_buffer_df[big_buffer_df['has_sample_point'] == True]

big_buffer_df = big_buffer_df[big_buffer_df['satelite_name'].str.contains('sentinel')]

big_buffer_df = big_buffer_df[big_buffer_df['cloud_cover(%)'] < 30]
big_buffer_df = big_buffer_df.sort_values('img_date', ascending=False)

best_image = big_buffer_df.iloc[0]

sample_row_sentinel = pd.concat([sample_row, best_image])

url = sample_row_sentinel['item'].assets['rendered_preview'].href
response = requests.get(url)

minx, miny, maxx, maxy = sample_row_sentinel['big_crop_bbox']
image = rioxarray.open_rasterio(pc.sign(
    sample_row_sentinel['item'].assets["visual"].href)).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )

image_array = image.to_numpy()

img_array_trans = np.transpose(image_array, axes=[1, 2, 0])
img_array_trans.shape

minx, miny, maxx, maxy = sample_row_sentinel['small_crop_bbox']
image = rioxarray.open_rasterio(pc.sign(
    sample_row_sentinel['item'].assets["visual"].href)).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )

image_array = image.to_numpy()
img_array_trans = np.transpose(image_array, axes=[1, 2, 0])

minx, miny, maxx, maxy = sample_row_sentinel['tiny_crop_bbox']
image = rioxarray.open_rasterio(
    pc.sign(sample_row_sentinel['item'].assets["visual"].href)).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )

image_array = image.to_numpy()
img_array_trans = np.transpose(image_array, axes=[1, 2, 0])

img_array_trans.shape

img_array_trans.min(), img_array_trans.max()

big_buffer_df = pd.DataFrame(pic_details)

big_buffer_df['has_sample_point'] = (
        (big_buffer_df.min_lat < sample_row.latitude)
        & (big_buffer_df.max_lat > sample_row.latitude)
        & (big_buffer_df.min_long < sample_row.longitude)
        & (big_buffer_df.max_long > sample_row.longitude)
    )

big_buffer_df = big_buffer_df[big_buffer_df['has_sample_point'] == True]

big_buffer_df = big_buffer_df[big_buffer_df['cloud_cover(%)'] <30]
big_buffer_df = big_buffer_df.sort_values('img_date', ascending=False)
big_buffer_df

best_ls_image = big_buffer_df.iloc[0]
best_ls_image

sample_row_ls = pd.concat([sample_row, best_image])
sample_row_ls

minx, miny, maxx, maxy = sample_row_ls['big_crop_bbox']

image = odc.stac.stac_load(
        [pc.sign(sample_row_ls['item'])], 
        bands=["red", "green", "blue"], 
        bbox=[minx, miny, maxx, maxy]
    ).isel(time=0)

image_array = image[["red", "green", "blue"]].to_array()
image_array.to_numpy().shape

img_array_trans = np.transpose(image_array.to_numpy(), axes=[1, 2, 0])
img_array_trans.shape

img_array_trans[0].max()

scaler = hab_functions.MinMaxScaler3D(feature_range=(0,255))
scaled_img = scaler.fit_transform(img_array_trans)
int_scaled_img = scaled_img.astype(int)
int_scaled_img.min(), int_scaled_img.max()

minx, miny, maxx, maxy = sample_row_ls['small_crop_bbox']
image = odc.stac.stac_load(
        [pc.sign(sample_row_ls['item'])], 
        bands=["red", "green", "blue"], 
        bbox=[minx, miny, maxx, maxy]
    ).isel(time=0)

image_array = image[["red", "green", "blue"]].to_array()
img_array_trans = np.transpose(image_array.to_numpy(), axes=[1, 2, 0])
scaled_img = scaler.fit_transform(img_array_trans)
int_scaled_img = scaled_img.astype(int)

minx, miny, maxx, maxy = sample_row_ls['tiny_crop_bbox']
image = odc.stac.stac_load(
        [pc.sign(sample_row_ls['item'])], 
        bands=["red", "green", "blue"], 
        bbox=[minx, miny, maxx, maxy]
    ).isel(time=0)

image_array = image[["red", "green", "blue"]].to_array()
img_array_trans = np.transpose(image_array.to_numpy(), axes=[1, 2, 0])
scaled_img = scaler.fit_transform(img_array_trans)
int_scaled_img = scaled_img.astype(int)

int_scaled_img.shape

int_scaled_img.min(), int_scaled_img.max()

sample_row = sat_train[sat_train['uid']=='garm'].iloc[0]

elev_coords = (sample_row.longitude, sample_row.latitude)
elev_coords

elev_bbox = sample_row['tiny_crop_bbox']
elev_bbox

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", 
    modifier=pc.sign_inplace
)

search = catalog.search(
    collections=["cop-dem-glo-30"],
    intersects={"type": "Point", "coordinates": elev_coords},
)
items = list(search.get_items())
print(f"Returned {len(items)} item.")

(minx, miny, maxx, maxy) = elev_bbox

signed_asset = pc.sign(items[0].assets["data"])
data = (
    rioxarray.open_rasterio(signed_asset.href).rio.clip_box(
        minx=minx,
        miny=miny,
        maxx=maxx,
        maxy=maxy,
        crs="EPSG:4326",
    )
    )

data.to_numpy().mean()

sample_row = sat_train[sat_train['uid'] == 'garm']

sample_row = sample_row[['uid', 'date', 'latitude', 'longitude']]
sample_row = sample_row.iloc[0]

sample_date = sample_row['date'] - timedelta(1)

sample_date = f'{sample_date:%Y%m%d}'

sample_lat = sample_row.latitude
sample_lon = sample_row.longitude

sector = "conus"
cycle = 18  # noon CST (times are in UTC)
forecast_hour = 1  # offset from cycle time
product = "wrfsfcf"  # 2D Pressure Levels
sample_date = sample_date  # August 8 2019

file_path = f"hrrr.t{cycle:02}z.{product}{forecast_hour:02}.grib2"
url = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{sample_date}/{sector}/{file_path}"
print(str(url))

file = tempfile.NamedTemporaryFile(prefix="delete_later_", 
                                   delete=False)

resp = requests.get(url)

with file as f:
    f.write(resp.content)

grbs = pygrib.open(file.name)

temp_grb = grbs.select(name='Temperature', level=0)[0]

lats, lons = temp_grb.latlons()
temps = temp_grb.values

lats_df = pd.DataFrame(lats)

lons_df = pd.DataFrame(lons)

lats_df = lats_df[(lats_df >(sample_lat-.05)) &
                  (lats_df < (sample_lat+.05))].dropna(how='all')

lons_df = lons_df[(lons_df >(sample_lon-.05)) &
                  (lons_df < (sample_lon+.05))].dropna(how='all')

lons_matches = lons_df.where(lons_df.notnull()).where(
    lats_df.notnull()).dropna(how='all').dropna(axis=1, how='all')


lats_df[[536, 537, 538]].dropna()

temp_results = pd.DataFrame(temps)[lons_matches.columns].loc[lons_matches.index]

temp_results = temp_results.mean().mean()

sample_row['temp_1day'] = temp_results

grbs.close()
file.close()
os.remove(file.name)

random_batch = sat_train.sample(500, random_state=14)

random_batch_features = hab_functions.get_sat_to_features(random_batch)

random_batch_features.head()

elev_random = random_batch[['uid','latitude', 'longitude', 'tiny_crop_bbox']]

elev_random['coords'] = list(zip(elev_random['longitude'], elev_random['latitude']))
elev_random = elev_random.drop(['longitude', 'latitude'], axis=1)

elev_random['elevation'] = np.NaN

elev_random.head()

catalog = Client.open(
"https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)

for index in range(len(elev_random)):

    row = elev_random.iloc[index]
    

    search = catalog.search(
    collections=["cop-dem-glo-30"],
    intersects={"type": "Point", "coordinates": row.coords},
    )
    items = list(search.get_items())


    (minx, miny, maxx, maxy) = row.tiny_crop_bbox

    signed_asset = pc.sign(items[0].assets["data"])
    data = (
        rioxarray.open_rasterio(signed_asset.href).rio.clip_box(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            crs="EPSG:4326",
        )
        )

    elev_random['elevation'].loc[row.name] = data.to_numpy().mean()

elevation_df = elev_random.copy()
elevation_df.head()


full_df_indexed = random_batch_features.reset_index().drop('index', axis=1)
full_df_indexed.tail()

model_df = full_df_indexed[['date', 'uid', 'region', 'latitude', 'longitude', 'season', 
                            'img_date', 'red_mean', 'red_median', 'red_max', 
                            'red_min','red_sum', 'red_product', 'green_mean', 
                            'green_median', 'green_max', 'green_min', 'green_sum',
                            'green_product', 'blue_mean', 'blue_median','blue_max',
                            'blue_min', 'blue_sum', 'blue_product', 'severity']]

model_df.isna().sum()

model_df[model_df['img_date'].isnull()].head()

missing_index_list = list(model_df[model_df['img_date'].isnull()].index)

model_df['missing_img_date'] = 0

for index in missing_index_list:
    model_df['missing_img_date'].loc[index] =1

model_df['missing_img_date'].value_counts()

model_df['img_date'] = model_df['img_date'].fillna(model_df['date'])

model_df.info()

model_df['date'] = model_df['date'].apply(lambda x: datetime.date(x))
model_df['latitude'] = model_df['latitude'].apply(lambda x: x.astype(float))
model_df['longitude'] = model_df['longitude'].apply(lambda x: x.astype(float))
model_df['severity'] = model_df['severity'].apply(lambda x: x.astype(int))
model_df['date'] = model_df['date'].apply(lambda x: x.toordinal())
model_df['img_date'] = model_df['img_date'].apply(lambda x: x.toordinal())
model_df['days_from_sat_to_sample'] = model_df['date'] - model_df['img_date']

model_df.head()


ohe = OneHotEncoder(sparse=False)
seasons = ohe.fit_transform(model_df[['season']])
cols = ohe.get_feature_names_out()

seasons = pd.DataFrame(seasons, columns=cols, index=model_df.index)
model_df = pd.concat([model_df.drop('season', axis=1), seasons], axis=1)

model_df.info()

