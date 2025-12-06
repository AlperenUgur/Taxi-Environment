# 🚖 6x6 Custom Taxi Environment – Q-Learning

Bu projede, 6x6 boyutunda özel olarak tasarlanmış bir grid ortamında hareket eden bir taksinin **Q-Learning** ile eğitilmesi amaçlanmaktadır.  
Taksi; yasak hücrelerden, duvarlardan ve hatalı eylemlerden kaçınarak yolcuyu alıp hedefe ulaştırmayı zamanla öğrenir.

---

## 🧩 1. Ortam Özeti

- Grid boyutu: 6×6
- Bazı hücreler yasak (kırmızı gösterilir) — taksi ve yolcu giremez.
- Hücreler arasında duvarlar vardır — taksi bu kenarlardan geçemez.
- Yolcu ve hedef konumu rastgele seçilir (yasak hücreler hariç).
- Tüm hücrelerden yolcu alınabilir ve bırakılabilir.

Amaç:  
Taksinin yolcuyu **en kısa ve güvenli şekilde** hedef konuma ulaştırması.

---

## 🧠 2. Kullanılan Yöntem – Q-Learning

Ajan, her durum–aksiyon çifti için bir Q değeri öğrenir.  
Güncelleme formülü şöyledir:

Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') – Q(s,a) ]


- **α (alpha):** öğrenme oranı  
- **γ (gamma):** geleceğe verilen önem  
- **r:** alınan ödül  
- **s’**: bir sonraki durum  

Ajan, ödülleri analiz ederek zamanla doğru politikayı öğrenir.

---
## 📦 3. Gerekli Kütüphaneler

  ```bash
  pip install gymnasium matplotlib imageio pillow numpy
  ```
---

## 📌 4. Durum (State) Temsili

Durum bilgisi şunlardan oluşur:

- **Taksi konumu:** `(taxi_row, taxi_col)`
- **Target konumu:** `(target_row, target_col)`  
  - Yolcu takside değilse: target = yolcu  
  - Yolcu taksideyse: target = varış noktası
- **Yolcu takside mi?** → `has_passenger` (0/1)

Toplam durum sayısı:  
**36 × 36 × 2 = 2.592 state**

Bu sıkıştırılmış state temsili hızlı ve stabil öğrenme sağlar.

---

## 🎛️ 5. Eylemler (Actions)

Ajan toplam 6 aksiyon gerçekleştirebilir:

| Kod | Aksiyon |
|-----|---------|
| 0   | Up      |
| 1   | Down    |
| 2   | Left    |
| 3   | Right   |
| 4   | Pickup  |
| 5   | Dropoff |

---

## 🎯 6. Ödül Sistemi

Aşağıdaki ödül sistemi ajanı verimli rotalar bulmaya yönlendirir:

| Durum | Ödül |
|-------|-------|
| Normal adım | -1 |
| Duvara / yasak hücreye yürümek | -2 |
| Yanlış pickup / dropoff | -4 |
| Doğru pickup | +5 |
| Doğru dropoff | +20 (bölüm biter) |
| Max adım aşımı | -10 |

---

## 🔄7. Salınım (Oscillation) Denetimi

Ajan bazen iki hücre arasında gidip gelebilir.  
Bunu engellemek için:

- Ters yönlü ardışık hareketlerde  
  → **-0.5 ceza** uygulanır.

Bu sayede taksi daha kararlı yollar seçer.

---

## 🏋️ 8. Eğitim Süreci

Eğitim başlatmak için:

python taxi_q_learning_6x6.py

İlk çalıştırmada:
```bash
DO_TRAIN = True
```
Q tablosu kaydedildikten sonra:
```bash
DO_TRAIN = False
```
yaparak tekrar eğitime gerek kalmadan ajanı kullanabilirsin.

---

## ▶️ 9. Canlı Animasyon (Matplotlib)
  ```bash

  watch_trained_agent(num_episodes=1, q_table=Q)

  ```
---
## 🎞️ 10. GIF Oluşturma
  ```bash

  save_run_as_gif(q_table=Q, max_steps=60, gif_name="taxi_episode.gif")

  ```




