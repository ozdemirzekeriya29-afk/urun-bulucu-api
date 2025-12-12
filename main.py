from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc

app = FastAPI(title="Akıllı Ürün Tanıma - REKABET MODU V8")

KLASOR = "urunler"

def goruntu_islee_ve_bul(gelen_resim_bytes):
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # Çözünürlük 1000px
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1000:
        oran = 1000 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1000, yeni_yukseklik))

    # Gri + Kontrast
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    img_gray = clahe.apply(img_gray)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    
    if des1 is None: return None, 0

    # --- DEĞİŞİKLİK BURADA: Sadece en iyiyi değil, TÜM adayları topluyoruz ---
    tum_adaylar = [] # Örnek: [('1704.jpg', 55), ('1705.jpg', 50)]

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
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                
                iyi_eslesmeler = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        iyi_eslesmeler.append(m)
                
                if len(iyi_eslesmeler) >= 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if mask is not None:
                        skor = sum(mask.ravel().tolist())
                        # Skoru listeye ekle (Skor, DosyaAdi)
                        tum_adaylar.append((skor, dosya.split(".")[0]))

            del db_img, kp2, des2
        except:
            pass

    gc.collect()

    # --- REKABET ANALİZİ ---
    if not tum_adaylar:
        return "Bulunamadı", 0
    
    # Skorlara göre büyükten küçüğe sırala
    tum_adaylar.sort(key=lambda x: x[0], reverse=True)
    
    en_iyi_skor = tum_adaylar[0][0]
    en_iyi_kod = tum_adaylar[0][1]
    
    # Eğer sadece 1 tane aday varsa ve skoru iyiyse döndür
    if len(tum_adaylar) == 1:
        return en_iyi_kod, en_iyi_skor

    # Eğer birden fazla aday varsa, 1. ile 2. arasındaki farka bak
    ikinci_en_iyi_skor = tum_adaylar[1][0]
    
    # KURAL: Birinci, ikinciden en az %20 daha iyi olmalı.
    # Değilse (puanlar yakınsa), sistem yanılıyordur. Reddet.
    fark_orani = en_iyi_skor / (ikinci_en_iyi_skor + 0.1) # 0'a bölünme hatası olmasın
    
    if fark_orani < 1.2: # %20 fark yoksa
        # "Kararsızım" mesajı döndür (Kod yerine None dönebiliriz veya özel mesaj)
        print(f"KARARSIZLIK: {en_iyi_kod} ({en_iyi_skor}) vs {tum_adaylar[1][1]} ({ikinci_en_iyi_skor})")
        return "Kararsiz", en_iyi_skor # Yanlış kod vermemek için özel durum
        
    return en_iyi_kod, en_iyi_skor

@app.get("/")
def ana_sayfa():
    return {"durum": "aktif", "mod": "REKABET MODU (Anti-Karışıklık)"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    # "Kararsiz" döndüyse kullanıcıya dürüst olalım
    if kod == "Kararsiz":
         return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": "Çok benzer ürünler var, netleyip tekrar çekin."}

    # Normal Baraj
    LIMIT = 15
    
    if skor >= LIMIT and kod != "Bulunamadı":
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "mesaj": "Bulundu"}
    else:
        mesaj = f"Yetersiz Veri ({skor})"
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": mesaj}
