# BERT ile Metin Sınıflandırma: IMDB Duygu Analizi

Bu proje, BERT (Bidirectional Encoder Representations from Transformers) modelini kullanarak IMDB film yorumları üzerinde duygu analizi gerçekleştiren kapsamlı bir makine öğrenmesi uygulamasıdır.

## 📋 İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Veri Seti](#veri-seti)
- [Model Mimarisi](#model-mimarisi)
- [Sonuçlar](#sonuçlar)
- [Teknolojiler](#teknolojiler)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## 🎯 Proje Hakkında

Bu proje, doğal dil işleme (NLP) alanında transformatör mimarilerinin gücünü göstermek amacıyla geliştirilmiştir. BERT modelini kullanarak IMDB film yorumlarının pozitif mi yoksa negatif mi olduğunu yüksek doğrulukla tahmin edebilen bir sistem oluşturulmuştur.

### Ana Hedefler
- BERT modelinin metin sınıflandırma performansını pratik bir örnekle göstermek
- Derin öğrenme projesinin tüm adımlarını kapsamlı şekilde uygulamak
- Hugging Face ekosisteminin etkin kullanımını örneklemek
- Modern NLP tekniklerini anlaşılır bir formatta sunmak

## ✨ Özellikler

- **Yüksek Performans**: %92+ test doğruluğu ve 0.975+ AUC skoru
- **Kapsamlı Değerlendirme**: Detaylı metrik hesaplamaları ve görselleştirmeler
- **Profesyonel İmplementasyon**: Epoch bazında model izleme ve otomatik en iyi model seçimi
- **Hız Optimizasyonu**: GPU desteği ile hızlı eğitim ve çıkarım
- **Görsel Analiz**: Karmaşıklık matrisi, ROC eğrisi ve eğitim grafikleri
- **Sağlam Altyapı**: Versiyon uyumluluğu sağlanmış kütüphane ortamı

## 🚀 Kurulum

### Gereksinimler

Bu proje Google Colab üzerinde çalışacak şekilde optimize edilmiştir. Aşağıdaki kütüphane versiyonları test edilmiştir:

```python
# Temel gereksinimler
numpy==1.26.4
datasets==3.6.0
transformers==4.48.3
scikit-learn
matplotlib
seaborn
torch
pandas
```

### Adım Adım Kurulum

1. **Colab Ortamını Hazırlayın**
   ```python
   # Çalışma zamanını tamamen sıfırlayın
   # Runtime -> Disconnect and delete runtime
   ```

2. **Kütüphaneleri Kurun**
   ```python
   # Notebook'un Hücre 1'ini çalıştırın
   !pip install numpy==1.26.4 -q
   !pip install datasets==3.6.0 -q
   !pip install transformers==4.48.3 -q
   !pip install scikit-learn matplotlib seaborn -q
   ```

3. **Çalışma Zamanını Yeniden Başlatın**
   ```
   Runtime -> Restart runtime
   ```

4. **GPU'yu Aktifleştirin**
   ```
   Runtime -> Change runtime type -> Hardware accelerator -> GPU
   ```

## 💻 Kullanım

### Temel Çalıştırma

1. Notebook'u Google Colab'da açın
2. Hücreleri sırasıyla çalıştırın:
   - **Hücre 1**: Kütüphane kurulumu
   - **Hücre 2**: Kütüphane importları ve versiyon kontrolü
   - **Hücre 3**: IMDB veri seti yükleme
   - **Hücre 4**: Veri setini eğitim/doğrulama/test olarak ayırma
   - **Hücre 5**: BERT tokenizer ile tokenizasyon
   - **Hücre 6**: Önceden eğitilmiş BERT modelini yükleme
   - **Hücre 7**: Model eğitimi
   - **Hücre 8**: Test seti değerlendirmesi
   - **Hücre 9**: Görselleştirmeler (Karmaşıklık matrisi, ROC eğrisi)
   - **Hücre 10**: Eğitim grafikleri
   - **Hücre 11**: Hız analizi

### Özelleştirme Seçenekleri

```python
# Eğitim parametrelerini değiştirmek için Hücre 7'deki değerleri düzenleyin
training_args = TrainingArguments(
    num_train_epochs=3,                    # Epoch sayısı
    per_device_train_batch_size=16,        # Eğitim batch boyutu
    per_device_eval_batch_size=32,         # Değerlendirme batch boyutu
    max_length=256,                        # Maksimum token uzunluğu
    # ... diğer parametreler
)
```

## 📊 Veri Seti

### IMDB Film Yorumları Veri Seti

- **Kaynak**: Hugging Face Datasets Hub
- **Boyut**: 50,000 etiketli yorum (25,000 eğitim + 25,000 test)
- **Sınıflar**: 
  - 0: Negatif duygu (olumsuz yorumlar)
  - 1: Pozitif duygu (olumlu yorumlar)
- **Denge**: Her sınıftan eşit sayıda örnek (dengeli veri seti)

### Veri Bölümü

- **Eğitim Seti**: 20,000 örnek (%80)
- **Doğrulama Seti**: 5,000 örnek (%20)  
- **Test Seti**: 25,000 örnek (orijinal test seti)

## 🏗️ Model Mimarisi

### BERT Modeli Detayları

- **Base Model**: `bert-base-uncased`
- **Parametre Sayısı**: ~110 milyon
- **Mimari**: 12 katmanlı Transformer enkoder
- **Sınıflandırma Başlığı**: 2 çıktılı (pozitif/negatif)
- **Tokenizer**: WordPiece tokenizasyon
- **Maksimum Uzunluk**: 256 token

### Eğitim Yapılandırması

```python
# Kullanılan eğitim parametreleri
- Epoch Sayısı: 3
- Eğitim Batch Boyutu: 16
- Değerlendirme Batch Boyutu: 32
- Öğrenme Oranı: 5e-5 (varsayılan)
- Warmup Steps: 500
- Weight Decay: 0.01
```

## 📈 Sonuçlar

### Test Seti Performansı

| Metrik | Değer |
|--------|--------|
| **Accuracy** | 92.04% |
| **Precision** | 91.94% |
| **Recall** | 92.15% |
| **F1-Score** | 92.05% |
| **Specificity** | 91.92% |
| **AUC** | 97.58% |

### Karmaşıklık Matrisi Sonuçları

|  | Tahmin Negatif | Tahmin Pozitif |
|--|----------------|----------------|
| **Gerçek Negatif** | 11,490 (TN) | 1,010 (FP) |
| **Gerçek Pozitif** | 981 (FN) | 11,519 (TP) |

### Performans Metrikleri

- **Eğitim Süresi**: ~22 dakika (NVIDIA L4 GPU)
- **Çıkarım Hızı**: ~3,242 örnek/saniye
- **Örnek Başına Çıkarım**: ~0.31 milisaniye

## 🛠️ Teknolojiler

### Ana Kütüphaneler

- **🤗 Transformers**: BERT modeli ve tokenizer
- **🤗 Datasets**: Veri seti yükleme ve işleme
- **🔥 PyTorch**: Derin öğrenme framework'ü
- **📊 Scikit-learn**: Metrik hesaplamaları
- **📈 Matplotlib/Seaborn**: Görselleştirmeler
- **🐼 Pandas/NumPy**: Veri manipülasyonu

### Geliştirme Ortamı

- **Platform**: Google Colab
- **GPU**: NVIDIA L4
- **Python**: 3.11+
- **CUDA**: 12.4

## 🎯 Gelecek Geliştirmeler

- [ ] Farklı BERT varyantları (RoBERTa, DeBERTa) ile karşılaştırma
- [ ] Hiperparametre optimizasyonu (Grid Search/Random Search)
- [ ] Cross-validation ile daha güvenilir değerlendirme
- [ ] Model distillation ile hız optimizasyonu
- [ ] Diğer dil modelleri ile karşılaştırmalı analiz
- [ ] Web arayüzü geliştirme
- [ ] Real-time duygu analizi API'si

## 🤝 Katkıda Bulunma

Bu projeye katkıda bulunmak istiyorsanız:

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

### Katkı Alanları

- 🐛 Bug raporları ve düzeltmeleri
- 📈 Performans iyileştirmeleri
- 📝 Dokümantasyon geliştirmeleri
- ✨ Yeni özellik önerileri
- 🧪 Test coverage artırma

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

## 🙏 Teşekkürler

- **Hugging Face** ekibine transformers ve datasets kütüphaneleri için
- **Google** ekibine BERT modelini açık kaynak yapması için
- **IMDB** veri setini oluşturan araştırmacılara
- **PyTorch** ve **Scikit-learn** topluluklarına


## 📞 İletişim

🐛 **Bug Report**: GitHub Issues kullanın  
💡 **Feature Request**: Discussions bölümünden önerinizi paylaşın  
📧 E-posta: [mehmetaksoy49@gmail.com]

- Pull Request ile katkıda bulunun
- Projeyi yıldızlamayı unutmayın! ⭐

---

**Not**: Bu proje eğitim amaçlı geliştirilmiştir ve akademik çalışmalarda referans olarak kullanılabilir.
