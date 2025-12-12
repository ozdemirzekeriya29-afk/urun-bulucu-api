from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc

app = FastAPI(title="Akıllı Ürün Tanıma - CROSS CHECK V9")

KLASOR = "urunler"

def goruntu_islee_ve_bul(gelen_resim_bytes):
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # Çözünürlük İyileştirmesi (1000px ideal)
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1000:
        oran = 1000 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1000, yeni_yukseklik))

    # Gri + CLAHE
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img_gray = clahe.apply(img_gray)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    
    if des1 is None: return None, 0

    # --- DEĞİŞİKLİK 1: BRUTE FORCE MATCHER (Kaba Kuvvet) ---
    # crossCheck=True: Eşleşmenin iki taraflı doğrulanmasını sağlar.
    # Yanlış eşleşmeleri inanılmaz derecede azaltır.
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    en_iyi_adaylar = [] # (Skor, Kod) listesi

    if not os.path.exists(KLASOR): return "VeritabaniYok", 0
    
    dosyalar = os.listdir(KLASOR)
    
    for dosya in dosyalar:
        if not dosya.endswith((".jpg", ".png", ".jpeg")): continue
        
        try:
            db_path = os.path.join(KLASOR, dosya)
            db_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
            if db_img is None: continue

            h, w = db_img.shape[:2]
            if w > 1000:
                scale = 1000 / w
                new_h = int(h * scale)
                db_img = cv2.resize(db_img, (1000, new_h))

            db_img = clahe.apply(db_img)
            kp2, des2 = sift.detectAndCompute(db_img, None)
            
            if des2 is not None and len(des2) > 10:
                # --- DEĞİŞİKLİK 2: ÇAPRAZ EŞLEŞTİRME ---
                matches = bf.match(des1, des2)
                
                # Eşleşmeleri mesafeye göre sırala (En kaliteliler üste)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # En iyi 50 eşleşmeyi al (Gürültüyü at)
                iyi_eslesmeler = matches[:50]

                # --- DEĞİŞİKLİK 3: GEOMETRİK SAĞLAMLIK ---
                if len(iyi_eslesmeler) >= 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    
                    # RANSAC ile geometriyi doğrula
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        # Maskeden dönen "inliers" (uyumlu nokta) sayısı bizim skorudur
                        skor = sum(mask.ravel().tolist())
                        
                        # Listeye ekle
                        if skor > 10: # En az 10 nokta geometriye uymalı
                            en_iyi_adaylar.append((skor, dosya.split(".")[0]))

            del db_img, kp2, des2
        except:
            pass

    gc.collect()

    # --- SONUÇ ANALİZİ ---
    if not en_iyi_adaylar:
        return "Bulunamadı", 0
    
    # Skorlara göre sırala (Büyükten küçüğe)
    en_iyi_adaylar.sort(key=lambda x: x[0], reverse=True)
    
    birinci_kod = en_iyi_adaylar[0][1]
    birinci_skor = en_iyi_adaylar[0][0]

    # Eğer tek aday varsa
    if len(en_iyi_adaylar) == 1:
        return birinci_kod, birinci_skor

    # İkinci ile fark analizi
    ikinci_skor = en_iyi_adaylar[1][0]
    
    # --- DEĞİŞİKLİK 4: FARK KURALI ---
    # Birinci, ikinciden en az 5 puan önde olmalı. 
    # (CrossCheck kullandığımız için puanlar düşük ama net olur. 5 puan büyük farktır).
    if birinci_skor > ikinci_skor + 5:
        return birinci_kod, birinci_skor
    else:
        # Fark çok azsa (Örn: Burgu 22, Spagetti 20), risk alma.
        print(f"KARARSIZLIK: {birinci_kod}({birinci_skor}) vs {en_iyi_adaylar[1][1]}({ikinci_skor})")
        return "Kararsiz", birinci_skor

@app.get("/")
def ana_sayfa():
    return {"durum": "aktif", "mod": "V9 - CROSS CHECK (Hatasız)"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    if kod == "Kararsiz":
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": "Çok benzer ürünler var. Lütfen isme odaklanıp kırpın."}

    # CrossCheck kullandığımız için puanlar daha düşük ama daha güvenilirdir.
    # 15 puan "Bu kesinlikle o" demek için yeterlidir.
    LIMIT = 15
    
    if skor >= LIMIT and kod != "Bulunamadı":
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "mesaj": "Bulundu"}
    else:
        mesaj = "Eşleşme Yok"
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": mesaj}
