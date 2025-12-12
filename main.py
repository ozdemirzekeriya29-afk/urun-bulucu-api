from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc

app = FastAPI(title="Akıllı Ürün Tanıma - AKAZE V10")

KLASOR = "urunler"

def goruntu_islee_ve_bul(gelen_resim_bytes):
    # 1. RESMİ OKU
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # Çözünürlük (AKAZE detay sever, 1000px iyidir)
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1000:
        oran = 1000 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1000, yeni_yukseklik))

    # Griye Çevir
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    
    # --- DEĞİŞİKLİK 1: AKAZE DETECTOR ---
    # SIFT yerine AKAZE kullanıyoruz. 
    # threshold: Detay hassasiyeti (Düşükse çok detay bulur).
    akaze = cv2.AKAZE_create(threshold=0.001)
    
    kp1, des1 = akaze.detectAndCompute(img_gray, None)
    
    if des1 is None: return None, 0

    # --- DEĞİŞİKLİK 2: HAMMING MESAFESİ ---
    # AKAZE binary (ikili) kod üretir. Bunu karşılaştırmak için
    # L2 değil, HAMMING kullanmalıyız. Çok daha hızlı ve kesindir.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    en_iyi_adaylar = []

    if not os.path.exists(KLASOR): return "VeritabaniYok", 0
    
    dosyalar = os.listdir(KLASOR)
    
    for dosya in dosyalar:
        if not dosya.endswith((".jpg", ".png", ".jpeg")): continue
        
        try:
            db_path = os.path.join(KLASOR, dosya)
            db_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
            if db_img is None: continue

            # Boyutlandırma
            h, w = db_img.shape[:2]
            if w > 1000:
                scale = 1000 / w
                new_h = int(h * scale)
                db_img = cv2.resize(db_img, (1000, new_h))

            kp2, des2 = akaze.detectAndCompute(db_img, None)
            
            if des2 is not None and len(des2) > 10:
                # Eşleştirme (CrossCheck Aktif)
                matches = bf.match(des1, des2)
                
                # Mesafeye göre sırala (En iyi eşleşmeler en üste)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Sadece mantıklı olanları al (Gürültüyü at)
                # Hamming mesafesi 50'den küçük olanlar "Çok İyi"dir.
                iyi_eslesmeler = [m for m in matches if m.distance < 60]

                # --- DEĞİŞİKLİK 3: GEOMETRİ DOĞRULAMA ---
                if len(iyi_eslesmeler) >= 12:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        skor = sum(mask.ravel().tolist())
                        
                        # Skor 15'ten büyükse listeye al
                        if skor >= 15:
                            en_iyi_adaylar.append((skor, dosya.split(".")[0]))

            del db_img, kp2, des2
        except:
            pass

    gc.collect()

    if not en_iyi_adaylar:
        return "Bulunamadı", 0
    
    # En yüksek skora göre sırala
    en_iyi_adaylar.sort(key=lambda x: x[0], reverse=True)
    
    birinci_skor = en_iyi_adaylar[0][0]
    birinci_kod = en_iyi_adaylar[0][1]

    # --- DEĞİŞİKLİK 4: REKABET ANALİZİ ---
    if len(en_iyi_adaylar) > 1:
        ikinci_skor = en_iyi_adaylar[1][0]
        # AKAZE skorları genelde daha sıkı olur.
        # Birinci ikinciden en az 5 puan önde olmalı.
        if birinci_skor < ikinci_skor + 5:
            return "Kararsiz", birinci_skor

    return birinci_kod, birinci_skor

@app.get("/")
def ana_sayfa():
    return {"durum": "aktif", "mod": "V10 - AKAZE (Keskin Göz)"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    if kod == "Kararsiz":
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": "Çok benzer ürünler var. Sadece isme odaklanıp kırpın."}

    # AKAZE ile 15-20 arası skor çok sağlamdır.
    LIMIT = 15
    
    if skor >= LIMIT and kod != "Bulunamadı":
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "mesaj": "Bulundu"}
    else:
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": "Eşleşme Yok"}
