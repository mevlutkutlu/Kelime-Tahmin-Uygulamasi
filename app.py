import streamlit as st
import numpy as np
import sounddevice as sd
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import os
from google.cloud import speech
from google.cloud import storage
from deep_translator import GoogleTranslator
from google.cloud import language_v1
from model_egit import model_guncelle
from pydub import AudioSegment

def test_baslat():
    

    # Eğitilmiş modeli yükleme
    model_kayit_yolu = r'C:\Users\Mevlit\Desktop\yazilim_sinama\model-yazilimsinama.pkl'
    model = joblib.load(model_kayit_yolu)

    # WAV dosyalarının bulunduğu klasör
    wav_dosyasi_klasoru = r'C:\Users\Mevlit\Desktop\yazilim_sinama\sesler-wav'

    # .wav uzantılı dosyaların isimlerini al
    sinif_isimleri = [
        os.path.splitext(dosya)[0] for dosya in os.listdir(wav_dosyasi_klasoru) if dosya.endswith(".wav")
    ]

    # Mikrofondan ses almak için gerekli parametreler
    saniye_basina_ornek = 44100  # Örnekleme hızı
    kanal_sayisi = 1  # Tek kanallı ses
    parca_suresi = 5  # Parça başına süre (saniye)
    parca_ornek = saniye_basina_ornek * parca_suresi  # 5 saniyelik örnek boyutu

    # "yazilim_sinama" dizini içinde "test" klasörü oluştur
    kayit_dizini = r"C:\Users\Mevlit\Desktop\yazilim_sinama\test"  # "test" klasörü yolu
    os.makedirs(kayit_dizini, exist_ok=True)  # Klasörü oluştur (zaten varsa hata vermez)

    # Tüm ses kaydı için dosya yolu
    tum_kayit_yolu = os.path.join(kayit_dizini, "tum_kayit.wav")
    
    parca_sayaci = 0

    # Tüm kaydı tek bir dosyada saklamak için SoundFile dosyası aç
    with sf.SoundFile(tum_kayit_yolu, mode='w', samplerate=saniye_basina_ornek, channels=kanal_sayisi) as tum_dosya:
        with sd.InputStream(samplerate=saniye_basina_ornek, channels=kanal_sayisi, dtype='float32') as stream:
            for _ in range(0, 60, parca_suresi):
                data = stream.read(parca_ornek)[0]  # 5 saniyelik veriyi oku
                tum_dosya.write(data)  # Tüm kaydı saklamak için dosyaya yaz
                parca_sayaci += 1

                # Her parçayı ayrı dosya olarak kaydet
                dosya_adi = os.path.join(kayit_dizini, f"parca_{parca_sayaci}.wav")
                sf.write(dosya_adi, data, samplerate=saniye_basina_ornek)

                # Parçayı işlemek için tahmin işlemi
                y, sr = librosa.load(dosya_adi, sr=saniye_basina_ornek)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
                mfcc = np.mean(mfcc.T, axis=0)  # Ortalama MFCC vektörü

                # Model üzerinden tahmin yapma
                tahmin_indeksi = model.predict(mfcc.reshape(1, -1))[0]
                tahmin_isim = sinif_isimleri[tahmin_indeksi]

                # Tahmini sonuç
                st.success(f"Tahmin edilen kişi: {tahmin_isim}")


    return True
            

def transkript_cikar():


    # Google Cloud kimlik dosyasını ayarlayın
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Mevlit\Desktop\yazilim_sinama\yazilimsinama-0648-ea2747e0ecd7.json"

    def ses_dosyasini_yukle(bucket_name, source_file_name, destination_blob_name):
        """Ses dosyasını Google Cloud Storage'a yükler."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)
        return f"gs://{bucket_name}/{destination_blob_name}"

    def transkript_cikart_uzun_ses(audio_uri):
        """Google Speech-to-Text ile uzun ses dosyasının transkriptini çıkarır."""
        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(uri=audio_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code="tr-TR"
        )

        # Uzun ses dosyasını işleme
        operation = client.long_running_recognize(config=config, audio=audio)
        st.info("Ses dosyası işleniyor, lütfen bekleyin...")
        response = operation.result(timeout=600)

        # Transkripti birleştirme
        transkript = ""
        for result in response.results:
            transkript += result.alternatives[0].transcript + "\n"
        return transkript

    # Ses dosyasının yolu ve Google Cloud Storage ayarları
    bucket_name = "ses-kayitlarim"
    source_file_name = r"C:\Users\Mevlit\Desktop\yazilim_sinama\test\tum_kayit.wav"
    destination_blob_name = "tum_kayit.wav"

    # Ses dosyasını yükleyip URI al
    audio_uri = ses_dosyasini_yukle(bucket_name, source_file_name, destination_blob_name)

    # Transkript oluştur ve ekrana yazdır
    transkript = transkript_cikart_uzun_ses(audio_uri)
    st.info(f"Transkript: {transkript}")
    

    # Transkripti bir dosyaya kaydet
    transkript_dosyasi = r"C:\Users\Mevlit\Desktop\yazilim_sinama\test\transkript.txt"
    with open(transkript_dosyasi, "w", encoding="utf-8") as f:
        f.write(transkript)
    st.success(f"Transkript kaydedildi: {transkript_dosyasi}")

    # Transkriptteki toplam kelime sayısını hesapla
    kelime_sayisi = len(transkript.split())
    st.info(f"Toplam Kelime Sayısı: {kelime_sayisi}")

    return True


def google_konu_analizi():
    # Transkripti dosyadan oku
    transkript_dosyasi = r"C:\Users\Mevlit\Desktop\yazilim_sinama\test\transkript.txt"
    with open(transkript_dosyasi, "r", encoding="utf-8") as f:
        transkript = f.read()

    # Deep Translator ile metni çevir
    translated = GoogleTranslator(source='tr', target='en').translate(transkript)
    st.text(f"Çevrilmiş Metin: {translated}")
    
    # Google Cloud Natural Language API ile analiz
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=translated, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_entities(document=document)

    # Konu analizi sonuçları
    st.text("Konu Analizi Sonuçları:")
    for entity in response.entities:
        st.text(f"Ad: {entity.name}, Tür: {entity.type_.name}, Önem Skoru: {entity.salience:.2f}")








# Yeni Kişi Ekle Fonksiyonu
def yeni_kisi_ekle():
    st.header("Yeni Kişi Ekle")

    # Kullanıcıdan giriş al
    kisi_adi = st.text_input("Yeni kişinin adını girin:", placeholder="Kişi adı girin ve Enter'a basın")

    if kisi_adi:  # Eğer bir isim girildiyse
        st.info(f"Kayıt işlemi başlıyor: {kisi_adi}")

        # Mikrofondan ses almak için gerekli parametreler
        saniye_basina_ornek = 44100  # Örnekleme hızı
        saniye = 90  # 90 saniyelik ses al
        kanal_sayisi = 1  # Tek kanallı ses

        st.info("Konuşmaya başlayın...")
        ses = sd.rec(int(saniye_basina_ornek * saniye), samplerate=saniye_basina_ornek, channels=kanal_sayisi, dtype='float32')
        sd.wait()

        # Ses dosyasını WAV olarak kaydetme
        kayit_yolu = fr'C:\Users\Mevlit\Desktop\yazilim_sinama\sesler-wav\{kisi_adi}.wav'
        sf.write(kayit_yolu, np.squeeze(ses), saniye_basina_ornek)
        st.success("Sesiniz başarıyla kaydedildi.")

        # Histogram ve spektrogram görselleştirme
        st.info("Ses dalga formu ve spektrogram hazırlanıyor...")
        y, sr = librosa.load(kayit_yolu, sr=None)

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # Ses dalga formunu çizin
        librosa.display.waveshow(y, sr=sr, ax=ax[0])
        ax[0].set_title("Ses Dalga Formu")
        ax[0].set_xlabel("Zaman (saniye)")
        ax[0].set_ylabel("Genlik")

        # Spektrogramı çizin
        S = librosa.stft(y)  # Short-time Fourier transform
        S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)  # Magnitüdü desibele dönüştür
        img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="hz", cmap="viridis", ax=ax[1])
        ax[1].set_title("Spektrogram")
        ax[1].set_xlabel("Zaman (saniye)")
        ax[1].set_ylabel("Frekans (Hz)")
        fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

        st.pyplot(fig)

        # Kaydedilen WAV dosyasını 2 saniyelik parçalara bölmek ve kaydetmek
        wav_file = kayit_yolu
        wav_dir = r'C:\Users\Mevlit\Desktop\yazilim_sinama\parcali-sesler-wav'  # Parçalanmış dosyaların kaydedileceği dizin
        os.makedirs(wav_dir, exist_ok=True)

        audio = AudioSegment.from_wav(wav_file)  # WAV dosyasını yükle
        chunks = audio[::2000]  # 2 saniyelik parçalar oluştur
        for i, chunk in enumerate(chunks):
            chunk.export(os.path.join(wav_dir, f"{kisi_adi}_{i}.wav"), format="wav")
        st.success("Ses dosyası parçalama işlemi tamamlandı.")

        # Kaydedilen kişinin MFCC özelliklerini çıkarmak
        ses_dizin = wav_dir
        mfcc_dizin = r'C:\Users\Mevlit\Desktop\yazilim_sinama\parcali-sesler-mfcc'
        os.makedirs(mfcc_dizin, exist_ok=True)

        n_mfcc = 128
        frame_length = 25  # milisaniye cinsinden
        frame_stride = 10  # milisaniye cinsinden

        for dosya_adı in os.listdir(ses_dizin):
            if dosya_adı.startswith(kisi_adi):
                dosya_yolu = os.path.join(ses_dizin, dosya_adı)
                ses, sr = librosa.load(dosya_yolu, sr=None)
                mfcc = librosa.feature.mfcc(y=ses, sr=sr, n_mfcc=n_mfcc, hop_length=int(frame_stride * sr / 1000),
                                             n_fft=int(frame_length * sr / 1000))
                mfcc_dosya_adı = dosya_adı.split('.')[0] + '.npy'
                mfcc_dosya_yolu = os.path.join(mfcc_dizin, mfcc_dosya_adı)
                np.save(mfcc_dosya_yolu, mfcc)

        st.success("MFCC özellikleri başarıyla oluşturuldu.") 
    model_guncelle()

# Canlı Test Yap Fonksiyonu
def canli_test_yap():
    st.header("Canlı Test Yap")
    st.write("Bu bölümde canlı ses testi yapabilirsiniz.")
    if st.button("Test Başlat"):
        st.info("Canlı test başlıyor!")
        tamamlandi = test_baslat()  # Test fonksiyonunu çağır
        if tamamlandi:
            tamamlandi2 = transkript_cikar()
            if tamamlandi2:
               tamamlandi3 = google_konu_analizi()
               if tamamlandi3:
                   st.success("Test sonuçlanmıştır")

            





# Streamlit Başlangıç
st.title("Kişi Tanıma ve Canlı Test Arayüzü")

# Menü Yapısı
menu = ["Ana Sayfa", "Yeni Kişi Ekle", "Canlı Test Yap"]
secim = st.sidebar.radio("Menü Seçin:", menu)

if secim == "Ana Sayfa":
    st.header("Hoş Geldiniz!")
    st.write("Bu uygulama, ses tanıma ve konu analizi işlemleri için geliştirilmiştir.")
    st.write("Lütfen sol menüden bir işlem seçin.")

elif secim == "Yeni Kişi Ekle":
    yeni_kisi_ekle()

elif secim == "Canlı Test Yap":
    canli_test_yap()




