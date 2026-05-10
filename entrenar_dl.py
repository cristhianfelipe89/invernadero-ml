# entrenar_dl.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Cargar datos y escalador existente
print("Cargando dataset y escalador local...")
df = pd.read_csv("datos/invernadero_cascada.csv")
X = df.drop("estado_riego", axis=1)
y = df["estado_riego"]

# Usamos la misma semilla que en tu SVM para consistencia
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = joblib.load('scaler_cascada.pkl')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Construir la Red Neuronal (ANN)
print("Construyendo modelo de Deep Learning...")
modelo_dl = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2), # Previene el sobreajuste
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid') # Salida binaria (0 o 1)
])

modelo_dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3. Entrenamiento con parada temprana
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Entrenando red neuronal...")
history = modelo_dl.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 4. Guardar modelo
modelo_dl.save('modelo_dl_cascada.h5')
print("✅ Modelo de Deep Learning guardado como 'modelo_dl_cascada.h5'")