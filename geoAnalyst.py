import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
features_df = pd.read_csv('features.csv')

# Функция для добавления признаков из features_df к df на основе ближайших соседей
def add_features(df, features_df):
    # Инициализация и обучение модели NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(features_df[['lat', 'lon']])
    
    distances, indices = nn.kneighbors(df[['lat', 'lon']])
    
    # Добавляем признаки из features_df к df
    features_to_add = features_df.drop(['lat', 'lon'], axis=1).iloc[indices.flatten()]
    df_extended = pd.concat([df.reset_index(drop=True), features_to_add.reset_index(drop=True)], axis=1)
    
    return df_extended

# Добавляем признаки к обучающему и тестовому наборам данных
train_df_extended = add_features(train_df, features_df)
test_df_extended = add_features(test_df, features_df)

# Подготовка данных для обучения модели
X = train_df_extended.drop(['id', 'lat', 'lon', 'score'], axis=1)
y = train_df_extended['score']

# Разбиение на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Оценка качества модели
y_pred_val = model.predict(X_val)
mae_val = mean_absolute_error(y_val, y_pred_val)

# Предсказание для тестового набора данных
X_test = test_df_extended.drop(['id', 'lat', 'lon'], axis=1)
y_pred_test = model.predict(X_test)

submission = pd.DataFrame({'id': test_df_extended['id'], 'score': y_pred_test})
submission.to_csv('submission.csv', index=False)
