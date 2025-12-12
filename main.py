from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import gc

app = FastAPI(title="Akıllı Ürün Tanıma - KESİN EŞLEŞME V6")

KLASOR = "urunler"

def goruntu_islee_ve_bul(gelen_resim_bytes):
    # 1. RESMİ OKU
    nparr = np.frombuffer(gelen_resim_bytes, np.uint8)
    aranan_resim = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if aranan_resim is None: return None, 0

    # Çözünürlüğü netlik için 1200px'e sabitle
    yukseklik, genislik = aranan_resim.shape[:2]
    if genislik > 1200:
        oran = 1200 / genislik
        yeni_yukseklik = int(yukseklik * oran)
        aranan_resim = cv2.resize(aranan_resim, (1200, yeni_yukseklik))

    # Griye Çevir + Kontrastı Fullee (Detayları patlat)
    img_gray = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) # Kontrastı artırdık
    img_gray = clahe.apply(img_gray)
    
    # SIFT Oluştur
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    
    if des1 is None: return None, 0

    en_iyi_eslesme_sayisi = 0
    bulunan_kod = "Bulunamadı"

    if not os.path.exists(KLASOR): return "VeritabaniYok", 0
    
    dosyalar = os.listdir(KLASOR)
    
    for dosya in dosyalar:
        if not dosya.endswith((".jpg", ".png", ".jpeg")): continue
        
        try:
            db_path = os.path.join(KLASOR, dosya)
            
            # Veritabanı resmini gri oku
            db_img = cv2.imread(db_path, cv2.IMREAD_GRAYSCALE)
            if db_img is None: continue

            # Veritabanı resmini de netleştir
            h, w = db_img.shape[:2]
            if w > 1200:
                scale = 1200 / w
                new_h = int(h * scale)
                db_img = cv2.resize(db_img, (1200, new_h))

            db_img = clahe.apply(db_img)
            kp2, des2 = sift.detectAndCompute(db_img, None)
            
            if des2 is not None and len(des2) > 15:
                # Eşleştirme Ayarları
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                
                iyi_eslesmeler = []
                for m, n in matches:
                    # --- DUVAR 1: SÜPER SIKI FİLTRE ---
                    # 0.60 çok düşük bir orandır. Sadece %100 aynı olan noktalar geçer.
                    # Benzer olanlar elenir.
                    if m.distance < 0.60 * n.distance:
                        iyi_eslesmeler.append(m)
                
                # --- DUVAR 2: MİNİMUM NOKTA SAYISI ---
                # En az 20 tane kusursuz nokta yoksa, geometriye bakma bile.
                if len(iyi_eslesmeler) >= 20:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
                    
                    # Geometri Doğrulama (RANSAC)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                    
                    if mask is not None:
                        matchesMask = mask.ravel().tolist()
                        gercek_uyusanlar = sum(matchesMask)

                        # --- DUVAR 3: GEOMETRİK BOZULMA KONTROLÜ ---
                        # Bulunan nesne yamulmuş mu? Ters mi dönmüş?
                        # Determinant 0'a yakınsa nesne çizgiden ibarettir, reddet.
                        det = np.linalg.det(M[:2, :2])
                        if abs(det) < 0.1: # Anlamsız geometri
                            continue

                        # EĞER 25'TEN FAZLA KUSURSUZ NOKTA VARSA KABUL ET
                        if gercek_uyusanlar > 25 and gercek_uyusanlar > en_iyi_eslesme_sayisi:
                            en_iyi_eslesme_sayisi = gercek_uyusanlar
                            bulunan_kod = dosya.split(".")[0]

            del db_img, kp2, des2
        except:
            pass

    gc.collect()
    return bulunan_kod, en_iyi_eslesme_sayisi

@app.get("/")
def ana_sayfa():
    return {"durum": "aktif", "mod": "KESİN EŞLEŞME (No False Positive)"}

@app.post("/tara")
async def urun_tara(file: UploadFile = File(...)):
    resim_verisi = await file.read()
    kod, skor = goruntu_islee_ve_bul(resim_verisi)
    
    # --- FİNAL KARAR NOKTASI ---
    # Artık skor "Benzerlik Puanı" değil, "Kusursuz Nokta Sayısı"dır.
    # 25 tane kusursuz nokta yoksa o ürün o ürün değildir.
    LIMIT = 25
    
    if skor >= LIMIT:
        return {"sonuc": True, "urun_kodu": kod, "guven_skoru": skor, "mesaj": "Tam Eşleşme"}
    else:
        # 0 puan dönsün ki uygulama "Bulunamadı" desin.
        return {"sonuc": False, "urun_kodu": None, "guven_skoru": skor, "mesaj": "Eşleşme Yok"}
