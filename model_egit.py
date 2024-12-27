def model_guncelle():
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import os
    import joblib

    # MFCC özelliklerinin bulunduğu dizin
    mfcc_dizin = r'C:\Users\Mevlit\Desktop\yazilim_sinama\parcali-sesler-mfcc'

    X = []
    y = []

    # MFCC dosyalarını yükleme
    for dosya_adı in os.listdir(mfcc_dizin):
        if dosya_adı.endswith('.npy'):
            dosya_yolu = os.path.join(mfcc_dizin, dosya_adı)
            mfcc = np.load(dosya_yolu)
            X.append(np.mean(mfcc, axis=1))  # Her dosya için ortalama MFCC vektörü
            y.append(dosya_adı.split('_')[0])  # Dosya adından etiket çıkarma

    X = np.array(X)
    y = np.array(y)

    # Etiketleri sayısal değerlere dönüştürme
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Veri kümesini eğitim ve test kümelerine ayırma
    X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # MLP modeli oluşturma ve eğitme
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_egitim, y_egitim)

    # Modelin doğruluğunu değerlendirme
    dogruluk = model.score(X_test, y_test)
    print(f"Model doğruluğu: {dogruluk}")

    print(X_egitim.shape)

    # Modeli diske kaydetme
    model_kayit_yolu = r'C:\Users\Mevlit\Desktop\yazilim_sinama\model-yazilimsinama.pkl'
    joblib.dump(model, model_kayit_yolu)
